"""Scenario definitions for the power grid environment.

Eight scenarios inspired by real grid events:
- Training: summer_peak, wind_drought, cold_snap, line_outage
- Test: cascading_failure, renewable_surplus, polar_vortex, price_spike_crisis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    """Complete scenario definition."""
    id: str
    scenario_type: str
    difficulty: str        # "normal", "hard", "expert"
    max_steps: int         # 96 = 24h, 288 = 72h
    description: str
    peak_demand_mw: float
    season: str            # Key into weather.SEASON_CONFIG
    start_hour: float
    generator_initial: dict[str, dict[str, Any]]
    battery_initial: dict[str, float]
    weather_config: dict[str, Any]
    events: list[dict[str, Any]]


# =============================================================================
# Default generator initial states
# =============================================================================

def _all_online_default() -> dict[str, dict[str, Any]]:
    """All generators online at typical dispatch point (~3450 MW thermal)."""
    return {
        "nuclear_1": {"status": "online", "output_mw": 1200.0},
        "coal_1":    {"status": "online", "output_mw": 500.0},
        "coal_2":    {"status": "online", "output_mw": 400.0},
        "ccgt_1":    {"status": "online", "output_mw": 500.0},
        "ccgt_2":    {"status": "online", "output_mw": 400.0},
        "peaker_1":  {"status": "online", "output_mw": 150.0},
        "peaker_2":  {"status": "online", "output_mw": 100.0},
        "peaker_3":  {"status": "offline", "output_mw": 0.0},
    }


def _high_demand_default() -> dict[str, dict[str, Any]]:
    """All generators online at maximum output for crisis scenarios."""
    return {
        "nuclear_1": {"status": "online", "output_mw": 1200.0},
        "coal_1":    {"status": "online", "output_mw": 500.0},
        "coal_2":    {"status": "online", "output_mw": 400.0},
        "ccgt_1":    {"status": "online", "output_mw": 500.0},
        "ccgt_2":    {"status": "online", "output_mw": 400.0},
        "peaker_1":  {"status": "online", "output_mw": 200.0},
        "peaker_2":  {"status": "online", "output_mw": 150.0},
        "peaker_3":  {"status": "online", "output_mw": 100.0},
    }


def _baseload_only() -> dict[str, dict[str, Any]]:
    """Low demand: only baseload units online."""
    return {
        "nuclear_1": {"status": "online", "output_mw": 1000.0},
        "coal_1":    {"status": "online", "output_mw": 300.0},
        "coal_2":    {"status": "online", "output_mw": 200.0},
        "ccgt_1":    {"status": "online", "output_mw": 225.0},
        "ccgt_2":    {"status": "offline", "output_mw": 0.0},
        "peaker_1":  {"status": "offline", "output_mw": 0.0},
        "peaker_2":  {"status": "offline", "output_mw": 0.0},
        "peaker_3":  {"status": "offline", "output_mw": 0.0},
    }


# =============================================================================
# Scenario Definitions
# =============================================================================

ALL_SCENARIOS: dict[str, Scenario] = {
    # ===== TRAINING =====

    "summer_peak": Scenario(
        id="summer_peak",
        scenario_type="summer_peak",
        difficulty="normal",
        max_steps=96,
        description=(
            "A hot summer day on a medium-sized utility grid serving ~2 million "
            "customers. Demand follows a typical summer profile peaking around "
            "3,500 MW (temperature-adjusted) near 5 PM. Solar helps during midday "
            "but the evening ramp from 3-6 PM is the main challenge as solar fades "
            "and AC load peaks. Your goal is to dispatch generation cost-effectively "
            "while maintaining reserves and frequency stability."
        ),
        peak_demand_mw=3500.0,
        season="summer",
        start_hour=0.0,
        generator_initial=_all_online_default(),
        battery_initial={"soc_mwh": 400.0},
        weather_config={
            "season": "summer",
            "events": [
                {"start_step": 60, "duration_steps": 8,
                 "event_type": "cloud_surge", "params": {"target_pct": 50.0}},
            ],
        },
        events=[],
    ),

    "wind_drought": Scenario(
        id="wind_drought",
        scenario_type="wind_drought",
        difficulty="hard",
        max_steps=96,
        description=(
            "A windy morning that turns dangerously calm. Wind generation starts at "
            "~400 MW (80% capacity factor) but drops to ~25 MW (5%) over 2 hours "
            "starting at 10 AM. You must detect the declining wind trend early and "
            "ramp thermal generators or start offline units to cover the 375 MW deficit. "
            "The battery can bridge the gap temporarily but cannot sustain for long. "
            "Failure to act proactively will cause reserve deficit and frequency drop."
        ),
        peak_demand_mw=3500.0,
        season="spring_windy",
        start_hour=6.0,
        generator_initial={
            "nuclear_1": {"status": "online", "output_mw": 1100.0},
            "coal_1":    {"status": "online", "output_mw": 350.0},
            "coal_2":    {"status": "online", "output_mw": 250.0},
            "ccgt_1":    {"status": "online", "output_mw": 300.0},
            "ccgt_2":    {"status": "online", "output_mw": 250.0},
            "peaker_1":  {"status": "offline", "output_mw": 0.0},
            "peaker_2":  {"status": "offline", "output_mw": 0.0},
            "peaker_3":  {"status": "offline", "output_mw": 0.0},
        },
        battery_initial={"soc_mwh": 480.0},
        weather_config={
            "season": "spring_windy",
            "events": [
                {"start_step": 16, "duration_steps": 8,
                 "event_type": "wind_ramp_down",
                 "params": {"from_pct": 80.0, "to_pct": 5.0}},
            ],
        },
        events=[],
    ),

    "cold_snap": Scenario(
        id="cold_snap",
        scenario_type="cold_snap",
        difficulty="hard",
        max_steps=96,
        description=(
            "An extreme cold snap hits the region, inspired by the February 2021 Texas "
            "winter storm. Temperature drops to -20C, pushing heating demand up sharply. "
            "At step 8 (2:00 AM), gas supply is curtailed and CCGT_1 is derated to "
            "250 MW. At step 20 (5:00 AM), peaker_3 trips due to frozen instrumentation. "
            "You must manage the shrinking generation fleet while demand rises with the "
            "evening peak. Strategic load shedding may be necessary to prevent total "
            "blackout."
        ),
        peak_demand_mw=3800.0,
        season="extreme_cold",
        start_hour=0.0,
        generator_initial=_high_demand_default(),
        battery_initial={"soc_mwh": 600.0},
        weather_config={
            "season": "extreme_cold",
            "events": [
                {"start_step": 8, "duration_steps": 16,
                 "event_type": "temp_plunge", "params": {"target_c": -25.0}},
            ],
        },
        events=[
            {"timestep": 8, "event_type": "generator_derate",
             "target": "ccgt_1", "params": {"capacity_mw": 250.0}},
            {"timestep": 20, "event_type": "generator_trip",
             "target": "peaker_3", "params": {}},
        ],
    ),

    "line_outage": Scenario(
        id="line_outage",
        scenario_type="line_outage",
        difficulty="hard",
        max_steps=96,
        description=(
            "A routine afternoon is disrupted by a transmission line failure. The main "
            "line from Zone A (baseload) to Zone B (load center) trips at step 12 "
            "(3:00 PM) due to equipment failure, severing 2,000 MW of transfer capacity. "
            "Zone B must rely on local generation (CCGTs, peakers) and the remaining "
            "transmission paths. Then at step 24 (6:00 PM), coal_1 trips offline, "
            "creating an N-1-1 contingency. You must redispatch generation across zones "
            "to respect transmission limits while maintaining reliability."
        ),
        peak_demand_mw=3500.0,
        season="summer",
        start_hour=0.0,
        generator_initial=_all_online_default(),
        battery_initial={"soc_mwh": 400.0},
        weather_config={"season": "summer", "events": []},
        events=[
            {"timestep": 12, "event_type": "line_outage",
             "target": "line_AB", "params": {}},
            {"timestep": 24, "event_type": "generator_trip",
             "target": "coal_1", "params": {}},
        ],
    ),

    # ===== TEST =====

    "cascading_failure": Scenario(
        id="cascading_failure",
        scenario_type="cascading_failure",
        difficulty="expert",
        max_steps=96,
        description=(
            "A cascading failure scenario inspired by the 2003 Northeast blackout. "
            "Starting from a moderately loaded system, line A->B trips at step 8 "
            "(2:00 AM). The system absorbs this but line B->C becomes heavily loaded. "
            "At step 16, line B->C trips from overload. At step 20, CCGT_2 trips. "
            "Frequency begins dropping. UFLS may activate automatically but you must "
            "manage controlled load shedding to prevent total blackout. Every second "
            "counts -- proactive load shedding is better than uncontrolled collapse."
        ),
        peak_demand_mw=3500.0,
        season="summer",
        start_hour=0.0,
        generator_initial=_all_online_default(),
        battery_initial={"soc_mwh": 500.0},
        weather_config={"season": "summer", "events": []},
        events=[
            {"timestep": 8, "event_type": "line_outage",
             "target": "line_AB", "params": {}},
            {"timestep": 16, "event_type": "line_outage",
             "target": "line_BC", "params": {}},
            {"timestep": 20, "event_type": "generator_trip",
             "target": "ccgt_2", "params": {}},
            {"timestep": 40, "event_type": "generator_trip",
             "target": "peaker_1", "params": {}},
        ],
    ),

    "renewable_surplus": Scenario(
        id="renewable_surplus",
        scenario_type="renewable_surplus",
        difficulty="hard",
        max_steps=96,
        description=(
            "A low-demand spring weekend with exceptional wind and solar output. Wind "
            "is at 90% capacity (~450 MW) and solar peaks at ~280 MW. Total demand is "
            "only 3,000 MW but nuclear + coal minimum generation is already ~1,160 MW. "
            "With 730 MW of renewables at midday, total minimum conventional + renewable "
            "exceeds demand. You must curtail renewables and/or charge the battery to "
            "avoid over-generation while managing frequency stability with reduced "
            "system inertia (fewer synchronous generators online)."
        ),
        peak_demand_mw=3000.0,
        season="spring_windy",
        start_hour=6.0,
        generator_initial=_baseload_only(),
        battery_initial={"soc_mwh": 200.0},
        weather_config={
            "season": "spring_windy",
            "events": [
                {"start_step": 20, "duration_steps": 12,
                 "event_type": "wind_ramp_up",
                 "params": {"from_pct": 90.0, "to_pct": 95.0}},
            ],
        },
        events=[],
    ),

    "polar_vortex": Scenario(
        id="polar_vortex",
        scenario_type="polar_vortex",
        difficulty="expert",
        max_steps=288,  # 72 hours (3 days)
        description=(
            "A multi-day polar vortex event inspired by the Texas 2021 crisis. Extreme "
            "cold persists for 72 hours with temperature-adjusted demand reaching "
            "~3,000 MW at midnight and higher during daytime heating peaks. Gas supply "
            "is progressively curtailed: CCGT_1 derated at step 8, CCGT_2 derated at "
            "step 24. Coal_2 trips at step 40 (frozen conveyor). Peaker_2 trips at "
            "step 60 (instrument freeze). Battery will be depleted quickly if not "
            "managed. You must plan strategic rolling blackouts over 3 days, cycling "
            "load shedding across zones to minimize total unserved energy while "
            "preventing cascading collapse. This is a marathon, not a sprint."
        ),
        peak_demand_mw=3500.0,
        season="extreme_cold",
        start_hour=0.0,
        generator_initial=_high_demand_default(),
        battery_initial={"soc_mwh": 700.0},
        weather_config={
            "season": "extreme_cold",
            "events": [
                {"start_step": 0, "duration_steps": 288,
                 "event_type": "temp_plunge", "params": {"target_c": -30.0}},
            ],
        },
        events=[
            {"timestep": 8, "event_type": "generator_derate",
             "target": "ccgt_1", "params": {"capacity_mw": 250.0}},
            {"timestep": 24, "event_type": "generator_derate",
             "target": "ccgt_2", "params": {"capacity_mw": 200.0}},
            {"timestep": 40, "event_type": "generator_trip",
             "target": "coal_2", "params": {}},
            {"timestep": 60, "event_type": "generator_trip",
             "target": "peaker_2", "params": {}},
        ],
    ),

    "price_spike_crisis": Scenario(
        id="price_spike_crisis",
        scenario_type="price_spike_crisis",
        difficulty="hard",
        max_steps=96,
        description=(
            "An extreme heat wave drives temperature-adjusted demand toward ~3,500 MW "
            "during the afternoon peak, straining the system to its limits. Temperature "
            "reaches 42C. All generators must run at maximum output. Market prices "
            "spike to $5,000/MWh. A demand surge event at step 16 adds 4% more load. "
            "There is intense political pressure: more than 2 consecutive hours (8 "
            "steps) of load shedding will trigger a political crisis penalty. You must "
            "balance cost against reliability, using every available resource including "
            "the battery to avoid or minimize blackouts during the afternoon peak."
        ),
        peak_demand_mw=3500.0,
        season="extreme_heat",
        start_hour=6.0,
        generator_initial=_high_demand_default(),
        battery_initial={"soc_mwh": 700.0},
        weather_config={
            "season": "extreme_heat",
            "events": [
                {"start_step": 16, "duration_steps": 20,
                 "event_type": "temp_spike", "params": {"target_c": 44.0}},
            ],
        },
        events=[
            {"timestep": 16, "event_type": "demand_surge",
             "target": "system", "params": {"factor": 1.04}},
        ],
    ),
}

TRAIN_SCENARIOS = ["summer_peak", "wind_drought", "cold_snap", "line_outage"]
TEST_SCENARIOS = ["cascading_failure", "renewable_surplus", "polar_vortex", "price_spike_crisis"]
SEEDS_PER_SCENARIO = 5


# =============================================================================
# Scenario Registry
# =============================================================================

class ScenarioRegistry:
    """Registry for accessing and listing scenarios."""

    @staticmethod
    def get(scenario_name: str) -> Scenario:
        if scenario_name not in ALL_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                             f"Available: {list(ALL_SCENARIOS.keys())}")
        return ALL_SCENARIOS[scenario_name]

    @staticmethod
    def list_tasks(split: str) -> list[dict]:
        if split == "train":
            scenarios = TRAIN_SCENARIOS
        elif split == "test":
            scenarios = TEST_SCENARIOS
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")

        tasks = []
        for sc_name in scenarios:
            sc = ALL_SCENARIOS[sc_name]
            for seed in range(SEEDS_PER_SCENARIO):
                task_id = f"{sc_name}_seed{seed}"
                tasks.append({
                    "id": task_id,
                    "scenario": sc_name,
                    "difficulty": sc.difficulty,
                    "max_steps": sc.max_steps,
                    "peak_demand_mw": sc.peak_demand_mw,
                    "season": sc.season,
                    "start_hour": sc.start_hour,
                    "generator_initial": sc.generator_initial,
                    "battery_initial": sc.battery_initial,
                    "weather_config": sc.weather_config,
                    "events": sc.events,
                    "seed": seed,
                })
        return tasks

    @staticmethod
    def list_splits() -> list[str]:
        return ["train", "test"]
