"""Power Grid Operator — OpenReward RL Environment.

A hyper-realistic power grid management environment where agents dispatch
generators, manage battery storage, handle renewable variability, and maintain
grid frequency across crisis scenarios inspired by real events.
"""

from __future__ import annotations

import copy
from typing import Any, List

from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

from generators import GeneratorStatus
from rewards import RewardCalculator
from scenarios import ScenarioRegistry
from simulation import GridSimulation


# =============================================================================
# Pydantic models for tool inputs
# =============================================================================

class TaskSpec(BaseModel):
    id: str
    scenario: str
    difficulty: str
    max_steps: int
    peak_demand_mw: float
    season: str
    start_hour: float
    generator_initial: dict
    battery_initial: dict
    weather_config: dict
    events: list
    seed: int = 0


class ObserveGridParams(BaseModel, extra="forbid"):
    """Read current grid state. Does NOT advance simulation time."""
    pass


class DispatchGeneratorsParams(BaseModel, extra="forbid"):
    """Set target output (MW) for one or more generators.
    units: dict mapping unit_id to target_output_mw.
    Example: {"nuclear_1": 1100, "ccgt_1": 450}
    """
    units: dict[str, float]


class ControlBatteryParams(BaseModel, extra="forbid"):
    """Control the grid-scale battery.
    action: "charge", "discharge", or "idle".
    power_mw: power level in MW (0-200).
    """
    action: str
    power_mw: float = 0.0


class ManageReservesParams(BaseModel, extra="forbid"):
    """Set a spinning reserve target (advisory — affects reserve score)."""
    reserve_mw: float


class ShedLoadParams(BaseModel, extra="forbid"):
    """Emergency load shedding.
    amount_mw: MW to shed.
    zone: "A", "B", "C", or "all".
    """
    amount_mw: float
    zone: str = "all"


class RestoreLoadParams(BaseModel, extra="forbid"):
    """Restore previously shed load.
    zone: "A", "B", "C", or "all".
    """
    zone: str = "all"


class StartGeneratorParams(BaseModel, extra="forbid"):
    """Begin startup sequence for an offline generator."""
    unit_id: str


class StopGeneratorParams(BaseModel, extra="forbid"):
    """Begin orderly shutdown of an online generator."""
    unit_id: str


class CurtailRenewableParams(BaseModel, extra="forbid"):
    """Curtail wind or solar output.
    source: "wind" or "solar".
    limit_mw: maximum output in MW.
    """
    source: str
    limit_mw: float


class AdvanceTimeParams(BaseModel, extra="forbid"):
    """Advance to the next 15-minute timestep without taking other actions."""
    pass


class SubmitLogParams(BaseModel, extra="forbid"):
    """Document reasoning. No simulation effect."""
    entry: str


# =============================================================================
# Main Environment Class
# =============================================================================

class PowerGridEnvironment(Environment):
    """Power grid operator management RL environment.

    Simulates a ~5,000 MW power system across crisis scenarios inspired
    by the Texas 2021 winter storm, 2003 Northeast blackout, and 2016
    South Australia blackout.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.sim: GridSimulation | None = None
        self.reward_calc: RewardCalculator | None = None
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.action_log: list[dict] = []

    async def setup(self) -> None:
        scenario = ScenarioRegistry.get(self.config.scenario)
        self.sim = GridSimulation(
            scenario_config={
                "max_steps": self.config.max_steps,
                "peak_demand_mw": self.config.peak_demand_mw,
                "season": self.config.season,
                "start_hour": self.config.start_hour,
                "generator_initial": self.config.generator_initial,
                "battery_initial": self.config.battery_initial,
                "weather_config": self.config.weather_config,
                "events": self.config.events,
            },
            seed=self.config.seed,
        )
        self.reward_calc = RewardCalculator(
            scenario_type=self.config.scenario,
            peak_demand_mw=self.config.peak_demand_mw,
        )
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.action_log = []

    async def teardown(self) -> None:
        self.sim = None
        self.reward_calc = None

    async def get_prompt(self) -> List[TextBlock]:
        scenario = ScenarioRegistry.get(self.config.scenario)
        prompt = f"""You are the chief grid operator for a medium-sized power utility serving ~2 million customers with a peak demand of {self.config.peak_demand_mw:.0f} MW.

## Current Scenario
{scenario.description}

## Your Fleet
| Unit | Type | Capacity | Min Output | Ramp Rate | Marginal Cost | Zone |
|------|------|----------|------------|-----------|---------------|------|
| nuclear_1 | Nuclear | 1,200 MW | 600 MW | 18 MW/min | ~$8/MWh | A |
| coal_1 | Coal | 500 MW | 200 MW | 10 MW/min | ~$25/MWh | A |
| coal_2 | Coal | 400 MW | 160 MW | 8 MW/min | ~$28/MWh | A |
| ccgt_1 | Gas CCGT | 500 MW | 225 MW | 30 MW/min | ~$35/MWh | B |
| ccgt_2 | Gas CCGT | 400 MW | 180 MW | 24 MW/min | ~$38/MWh | B |
| peaker_1 | Gas Peaker | 200 MW | 50 MW | 30 MW/min | ~$55/MWh | B |
| peaker_2 | Gas Peaker | 150 MW | 37 MW | 22.5 MW/min | ~$60/MWh | C |
| peaker_3 | Gas Peaker | 100 MW | 25 MW | 15 MW/min | ~$65/MWh | C |

**Battery Storage**: 200 MW / 800 MWh, 85% round-trip efficiency, SoC limits 10-90%
**Wind Farm**: 500 MW nameplate (Zone C)
**Solar Farm**: 300 MW nameplate (Zone B/C)

## Transmission Network (3 zones)
- Line A->B: 2,000 MW capacity (baseload zone to load center)
- Line B->C: 1,500 MW capacity
- Line A->C: 800 MW capacity
- Zone demand split: A=30%, B=50%, C=20%

## Key Physics
- **Frequency**: 60 Hz nominal. Governor droop 5%, deadband 36 mHz.
- **UFLS**: Automatic load shedding at 59.5 Hz (10%), 59.1 Hz (+10%), 58.7 Hz (+10%).
- **Blackout**: Frequency below 57.5 Hz = total system collapse.
- **Reserves**: Must maintain spinning reserve >= max(largest online unit, 3% load + 3% gen).
- **Ramp limits**: Generators cannot change output faster than their ramp rate.
- **Min up/down time**: Nuclear=24h, Coal=8h, CCGT=4h/2h, Peaker=30min/15min.

## Available Actions
1. **observe_grid** — Read full grid state (FREE, no time advance). Call this first!
2. **dispatch_generators** — Set MW targets for generators (respects ramp limits).
3. **control_battery** — Charge/discharge/idle the battery.
4. **manage_reserves** — Set spinning reserve target (advisory).
5. **shed_load** — Emergency load shedding by zone. Last resort!
6. **restore_load** — Restore previously shed load.
7. **start_generator** — Begin startup of an offline unit.
8. **stop_generator** — Shut down an online unit.
9. **curtail_renewable** — Limit wind/solar output.
10. **advance_time** — Move to next 15-min timestep.
11. **submit_log** — Document your reasoning (no simulation effect).

## Important Notes
- Each action tool call (except observe_grid and submit_log) advances time by 15 minutes.
- You have {self.config.max_steps} timesteps ({self.config.max_steps * 15 / 60:.0f} hours).
- Blackouts are catastrophic: -1.0 terminal reward.
- Your reward depends on: reliability (40%), cost efficiency (25%), frequency stability (15%), reserve adequacy (10%), renewable utilization (10%).

Begin by calling observe_grid to assess the current situation."""

        return [TextBlock(text=prompt)]

    # =========================================================================
    # Helper: advance simulation and return output
    # =========================================================================

    def _advance_time_and_get_output(self, action_name: str,
                                     action_detail: str) -> ToolOutput:
        assert self.sim is not None
        assert self.reward_calc is not None

        self.sim.advance()
        self.step_count += 1

        state = self.sim.state
        reward = self.reward_calc.step_reward(state, dt_hours=0.25)
        self.cumulative_reward += reward

        is_terminal, reason = self.reward_calc.is_terminal(
            state, self.step_count, self.config.max_steps)

        if is_terminal:
            terminal_r = self.reward_calc.terminal_reward(state)
            reward += terminal_r
            self.cumulative_reward += terminal_r

        # Format display
        display = self.sim.format_status()

        summary = f"\n--- ACTION: {action_name} ---\n{action_detail}\n"
        summary += f"Step Reward: {reward:+.3f} | Cumulative: {self.cumulative_reward:+.3f}\n"

        if is_terminal:
            summary += f"\n*** EPISODE ENDED: {reason.upper().replace('_', ' ')} ***\n"
            if reason == "completed":
                summary += "Scenario completed successfully.\n"
            elif reason == "blackout":
                summary += "CATASTROPHIC FAILURE - Total grid collapse.\n"
            elif reason == "frequency_collapse":
                summary += "CATASTROPHIC FAILURE - Frequency collapsed below 57.5 Hz.\n"

        # Log
        self.action_log.append({
            "step": self.step_count,
            "action": action_name,
            "detail": action_detail,
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "terminal": is_terminal,
            "reason": reason if is_terminal else None,
        })

        breakdown = self.reward_calc.get_reward_breakdown(state)

        return ToolOutput(
            metadata={
                "step": self.step_count,
                "max_steps": self.config.max_steps,
                "reward": round(reward, 4),
                "cumulative_reward": round(self.cumulative_reward, 4),
                "terminal": is_terminal,
                "reason": reason if is_terminal else None,
                "reward_breakdown": breakdown,
                "frequency_hz": round(state.frequency_hz, 3),
                "total_demand_mw": round(state.total_demand_mw, 1),
                "total_generation_mw": round(state.total_generation_mw, 1),
                "total_load_shed_mw": round(state.total_load_shed_mw, 1),
                "battery_soc_pct": round(state.battery_soc_pct, 1),
            },
            blocks=[TextBlock(text=summary + "\n" + display)],
            reward=reward,
            finished=is_terminal,
        )

    # =========================================================================
    # Tools
    # =========================================================================

    @tool
    async def observe_grid(self, params: ObserveGridParams) -> ToolOutput:
        """Read the current grid state. Does NOT advance simulation time.
        Returns: system frequency, demand, generation by unit, battery status,
        reserves, weather, transmission line flows, and costs.
        """
        assert self.sim is not None
        display = self.sim.format_status()
        state_dict = self.sim.get_state_dict()

        return ToolOutput(
            metadata=state_dict,
            blocks=[TextBlock(text=display)],
            reward=0.0,
            finished=False,
        )

    @tool
    async def dispatch_generators(self, params: DispatchGeneratorsParams) -> ToolOutput:
        """Set target output (MW) for one or more generators. Targets are subject
        to ramp rate limits, minimum/maximum output constraints, and generator
        status. Units not specified keep their current targets.
        """
        assert self.sim is not None
        messages = []
        for uid, target in params.units.items():
            gen = self.sim.generators.get(uid)
            if gen is None:
                messages.append(f"Unknown unit: {uid}")
                continue
            ok, msg = gen.set_target(target)
            messages.append(msg)

        detail = "Dispatch: " + " | ".join(messages)
        return self._advance_time_and_get_output("dispatch_generators", detail)

    @tool
    async def control_battery(self, params: ControlBatteryParams) -> ToolOutput:
        """Control the grid-scale battery storage system.
        action: 'charge' (grid -> battery), 'discharge' (battery -> grid), 'idle'.
        power_mw: power level (0-200 MW).
        """
        assert self.sim is not None
        dt_hours = self.sim.dt_min / 60.0

        if params.action == "charge":
            actual = self.sim.battery.charge(params.power_mw, dt_hours)
            detail = f"Battery charging at {actual:.1f} MW (requested {params.power_mw:.1f})"
        elif params.action == "discharge":
            actual = self.sim.battery.discharge(params.power_mw, dt_hours)
            detail = f"Battery discharging at {actual:.1f} MW (requested {params.power_mw:.1f})"
        elif params.action == "idle":
            self.sim.battery.set_idle()
            detail = "Battery set to idle."
        else:
            return ToolOutput(
                metadata={"error": f"Unknown action: {params.action}"},
                blocks=[TextBlock(text=f"Error: Unknown action '{params.action}'. "
                                       f"Use 'charge', 'discharge', or 'idle'.")],
                reward=0.0,
                finished=False,
            )

        return self._advance_time_and_get_output("control_battery", detail)

    @tool
    async def manage_reserves(self, params: ManageReservesParams) -> ToolOutput:
        """Set a spinning reserve target (advisory). This helps track whether
        your reserve position is adequate but does not directly change dispatch.
        The reserve adequacy component of your reward depends on actual reserves
        vs. the system requirement.
        """
        assert self.sim is not None
        detail = f"Reserve target set to {params.reserve_mw:.0f} MW."
        return self._advance_time_and_get_output("manage_reserves", detail)

    @tool
    async def shed_load(self, params: ShedLoadParams) -> ToolOutput:
        """Emergency load shedding. This is a last resort to prevent frequency
        collapse. Shedding load disconnects customers and incurs heavy reliability
        penalties. More than 8 consecutive steps (2 hours) of shedding triggers
        additional political crisis penalty.
        zone: 'A', 'B', 'C', or 'all'.
        """
        assert self.sim is not None
        amount = max(0.0, params.amount_mw)

        if params.zone == "all":
            per_zone = amount / len(self.sim.zones)
            for z in self.sim.zones.values():
                z.shed_mw = per_zone
        else:
            zone = self.sim.zones.get(params.zone)
            if zone is None:
                return ToolOutput(
                    metadata={"error": f"Unknown zone: {params.zone}"},
                    blocks=[TextBlock(text=f"Error: Unknown zone '{params.zone}'. Use 'A', 'B', 'C', or 'all'.")],
                    reward=0.0,
                    finished=False,
                )
            zone.shed_mw = amount

        total_shed = sum(z.shed_mw for z in self.sim.zones.values())
        detail = f"Load shedding: {total_shed:.0f} MW (zone: {params.zone})"
        return self._advance_time_and_get_output("shed_load", detail)

    @tool
    async def restore_load(self, params: RestoreLoadParams) -> ToolOutput:
        """Restore previously shed load. Should be done when generation capacity
        is sufficient to serve demand.
        """
        assert self.sim is not None

        if params.zone == "all":
            for z in self.sim.zones.values():
                z.shed_mw = 0.0
            detail = "All load restored across all zones."
        else:
            zone = self.sim.zones.get(params.zone)
            if zone is None:
                return ToolOutput(
                    metadata={"error": f"Unknown zone: {params.zone}"},
                    blocks=[TextBlock(text=f"Error: Unknown zone '{params.zone}'.")],
                    reward=0.0,
                    finished=False,
                )
            zone.shed_mw = 0.0
            detail = f"Load restored in zone {params.zone}."

        return self._advance_time_and_get_output("restore_load", detail)

    @tool
    async def start_generator(self, params: StartGeneratorParams) -> ToolOutput:
        """Begin startup sequence for an offline generator. Startup times vary:
        nuclear (12-48h), coal (2-12h), CCGT (0.5-4h), peaker (5-20min).
        Hot starts are faster if the unit was recently shut down.
        """
        assert self.sim is not None
        gen = self.sim.generators.get(params.unit_id)
        if gen is None:
            return ToolOutput(
                metadata={"error": f"Unknown unit: {params.unit_id}"},
                blocks=[TextBlock(text=f"Error: No generator '{params.unit_id}' exists.")],
                reward=0.0,
                finished=False,
            )
        ok, msg = gen.begin_startup()
        return self._advance_time_and_get_output("start_generator", msg)

    @tool
    async def stop_generator(self, params: StopGeneratorParams) -> ToolOutput:
        """Begin orderly shutdown of an online generator. Respects minimum
        up-time constraints. The generator will ramp down to zero.
        """
        assert self.sim is not None
        gen = self.sim.generators.get(params.unit_id)
        if gen is None:
            return ToolOutput(
                metadata={"error": f"Unknown unit: {params.unit_id}"},
                blocks=[TextBlock(text=f"Error: No generator '{params.unit_id}' exists.")],
                reward=0.0,
                finished=False,
            )
        ok, msg = gen.begin_shutdown()
        return self._advance_time_and_get_output("stop_generator", msg)

    @tool
    async def curtail_renewable(self, params: CurtailRenewableParams) -> ToolOutput:
        """Curtail wind or solar output to a maximum MW level. Use when
        over-generation threatens frequency stability. Curtailment reduces
        your renewable utilization score.
        """
        assert self.sim is not None
        if params.source == "wind":
            msg = self.sim.wind.curtail(params.limit_mw)
        elif params.source == "solar":
            msg = self.sim.solar.curtail(params.limit_mw)
        else:
            return ToolOutput(
                metadata={"error": f"Unknown source: {params.source}"},
                blocks=[TextBlock(text=f"Error: Unknown source '{params.source}'. Use 'wind' or 'solar'.")],
                reward=0.0,
                finished=False,
            )
        return self._advance_time_and_get_output("curtail_renewable", msg)

    @tool
    async def advance_time(self, params: AdvanceTimeParams) -> ToolOutput:
        """Advance to the next 15-minute timestep without taking any other
        action. Use when the current dispatch is satisfactory and you want
        to monitor the system.
        """
        assert self.sim is not None
        return self._advance_time_and_get_output(
            "advance_time", "Monitoring — no operational action taken.")

    @tool
    async def submit_log(self, params: SubmitLogParams) -> ToolOutput:
        """Document your reasoning and observations. No simulation effect.
        Use this to record your analysis, plans, or concerns.
        """
        self.action_log.append({
            "step": self.step_count,
            "action": "log",
            "detail": params.entry,
        })
        return ToolOutput(
            metadata={"logged": True},
            blocks=[TextBlock(text=f"Log recorded: {params.entry}")],
            reward=0.0,
            finished=False,
        )

    # =========================================================================
    # Class methods for task/split enumeration
    # =========================================================================

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return ScenarioRegistry.list_tasks(split)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ScenarioRegistry.list_splits()
