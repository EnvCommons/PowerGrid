"""Grid simulation engine coupling generators, weather, frequency, demand, and transmission.

This is the core physics engine for the power grid environment. It manages the
complete state of a ~5,000 MW power system across 15-minute timesteps.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from generators import (
    GENERATOR_FLEET, INERTIA_CONSTANTS, BatteryStorage, GeneratorSpec,
    GeneratorStatus, GeneratorUnit, RenewableSource,
)
from weather import (
    WeatherModel, WeatherState, calculate_solar_output, calculate_wind_output,
)


# =============================================================================
# Constants
# =============================================================================

NOMINAL_FREQ_HZ = 60.0
GOVERNOR_DEADBAND_HZ = 0.036  # FERC Order 842 max: +/- 36 mHz
GOVERNOR_DROOP = 0.05          # FERC Order 842 / NERC: 5% max droop

# Under-frequency load shedding stages (frequency_hz, fraction_of_load_shed)
# Aligned with NERC PRC-006-5; total 30% shed across 3 stages.
UFLS_STAGES = [
    (59.5, 0.10),  # Stage 1: 10% at 59.5 Hz
    (59.1, 0.10),  # Stage 2: +10% at 59.1 Hz
    (58.7, 0.10),  # Stage 3: +10% at 58.7 Hz
]
# Blackout threshold: 0.5 Hz above IEEE C37.106 turbine damage at 57.0 Hz
BLACKOUT_FREQ_HZ = 57.5

# Transmission network (3-zone model)
DEFAULT_LINES = [
    {"line_id": "line_AB", "from_zone": "A", "to_zone": "B", "capacity_mw": 2000.0},
    {"line_id": "line_BC", "from_zone": "B", "to_zone": "C", "capacity_mw": 1500.0},
    {"line_id": "line_AC", "from_zone": "A", "to_zone": "C", "capacity_mw": 800.0},
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TransmissionLine:
    line_id: str
    from_zone: str
    to_zone: str
    capacity_mw: float
    current_flow_mw: float = 0.0
    status: str = "online"  # "online", "tripped"

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "capacity_mw": self.capacity_mw,
            "current_flow_mw": round(self.current_flow_mw, 1),
            "status": self.status,
            "loading_pct": round(abs(self.current_flow_mw) / max(self.capacity_mw, 1) * 100, 1),
        }


@dataclass
class LoadZone:
    zone_id: str
    demand_fraction: float  # Fraction of total demand in this zone
    shed_mw: float = 0.0

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "demand_fraction": self.demand_fraction,
            "shed_mw": round(self.shed_mw, 1),
        }


@dataclass
class GridState:
    """Complete snapshot of grid state at a point in time."""
    timestep: int = 0
    elapsed_minutes: float = 0.0
    hour_of_day: float = 0.0

    # Frequency
    frequency_hz: float = 60.0

    # Aggregates
    total_generation_mw: float = 0.0
    total_demand_mw: float = 0.0
    total_renewable_mw: float = 0.0
    wind_output_mw: float = 0.0
    solar_output_mw: float = 0.0

    # Battery
    battery_power_mw: float = 0.0  # Positive = discharging
    battery_soc_pct: float = 50.0

    # Load shedding
    total_load_shed_mw: float = 0.0
    ufls_shed_mw: float = 0.0  # Automatic UFLS shed (separate from manual)
    manual_shed_mw: float = 0.0

    # Reserves
    spinning_reserve_mw: float = 0.0
    required_reserve_mw: float = 0.0

    # Costs
    generation_cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    lmp_usd_per_mwh: float = 0.0

    # Weather snapshot
    temperature_c: float = 25.0
    wind_speed_m_s: float = 8.0
    cloud_cover_pct: float = 20.0

    # Renewable curtailment
    wind_curtailed_mw: float = 0.0
    solar_curtailed_mw: float = 0.0

    # Flags
    ufls_triggered: bool = False
    blackout: bool = False


@dataclass
class ScheduledEvent:
    """An event that fires at a specific timestep."""
    timestep: int
    event_type: str  # "generator_trip", "line_outage", "generator_derate",
                     # "gas_curtail", "demand_surge", "line_restore"
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    fired: bool = False


# =============================================================================
# Demand model
# =============================================================================

# Summer load profile (hourly fraction of peak, interpolated)
# Peak at 17:00, trough at 03:00
SUMMER_LOAD_CURVE = {
    0: 0.58, 1: 0.56, 2: 0.55, 3: 0.55, 4: 0.56, 5: 0.58,
    6: 0.65, 7: 0.72, 8: 0.78, 9: 0.82, 10: 0.85, 11: 0.88,
    12: 0.90, 13: 0.92, 14: 0.95, 15: 0.97, 16: 0.99, 17: 1.00,
    18: 0.97, 19: 0.92, 20: 0.85, 21: 0.78, 22: 0.70, 23: 0.63,
}

# Winter load profile (double peak: morning 7-9, evening 17-19)
WINTER_LOAD_CURVE = {
    0: 0.63, 1: 0.61, 2: 0.60, 3: 0.60, 4: 0.62, 5: 0.68,
    6: 0.76, 7: 0.85, 8: 0.88, 9: 0.84, 10: 0.80, 11: 0.78,
    12: 0.76, 13: 0.78, 14: 0.80, 15: 0.83, 16: 0.88, 17: 0.93,
    18: 0.95, 19: 0.95, 20: 0.90, 21: 0.83, 22: 0.75, 23: 0.68,
}


def interpolate_load_curve(hour: float, curve: dict[int, float]) -> float:
    """Linear interpolation of hourly load curve."""
    h_lo = int(hour) % 24
    h_hi = (h_lo + 1) % 24
    frac = hour - int(hour)
    return curve[h_lo] * (1 - frac) + curve[h_hi] * frac


# =============================================================================
# Grid Simulation
# =============================================================================

class GridSimulation:
    """Core simulation engine for the power grid."""

    def __init__(self, scenario_config: dict, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.dt_min = 15.0  # 15 minutes per timestep
        self.max_steps = scenario_config.get("max_steps", 96)

        # Peak demand
        self.peak_demand_mw = scenario_config.get("peak_demand_mw", 5000.0)
        self.season = scenario_config.get("season", "summer")

        # Initialize generators
        gen_init = scenario_config.get("generator_initial", {})
        self.generators: dict[str, GeneratorUnit] = {}
        for uid, spec in GENERATOR_FLEET.items():
            gi = gen_init.get(uid, {})
            status = gi.get("status", "offline")
            output = gi.get("output_mw", 0.0)
            derated = gi.get("derated_capacity_mw", None)
            self.generators[uid] = GeneratorUnit(spec, status, output, derated)

        # Battery
        bat_init = scenario_config.get("battery_initial", {})
        self.battery = BatteryStorage(
            initial_soc_mwh=bat_init.get("soc_mwh", 400.0),
        )

        # Renewables
        self.wind = RenewableSource("wind", 500.0)
        self.solar = RenewableSource("solar", 300.0)

        # Weather
        weather_config = scenario_config.get("weather_config", {})
        weather_season = weather_config.get("season", self.season)
        weather_events = weather_config.get("events", [])
        start_hour = scenario_config.get("start_hour", 0.0)
        self.weather = WeatherModel(
            season=weather_season, seed=seed,
            start_hour=start_hour, events=weather_events,
        )

        # Transmission
        self.lines: list[TransmissionLine] = []
        for ld in scenario_config.get("lines", DEFAULT_LINES):
            self.lines.append(TransmissionLine(**ld))

        # Load zones (A: 30%, B: 50%, C: 20%)
        zone_config = scenario_config.get("zones", {})
        self.zones: dict[str, LoadZone] = {
            "A": LoadZone("A", zone_config.get("A", 0.30)),
            "B": LoadZone("B", zone_config.get("B", 0.50)),
            "C": LoadZone("C", zone_config.get("C", 0.20)),
        }

        # Scheduled events
        self.events: list[ScheduledEvent] = []
        for e in scenario_config.get("events", []):
            self.events.append(ScheduledEvent(
                timestep=e["timestep"],
                event_type=e["event_type"],
                target=e["target"],
                params=e.get("params", {}),
            ))

        # State
        self.state = GridState()
        self.state.hour_of_day = start_hour

        # Compute initial state
        self._update_weather_and_renewables()
        self._update_demand()
        self._update_aggregates()

    # -----------------------------------------------------------------
    # Core simulation step
    # -----------------------------------------------------------------

    def advance(self) -> None:
        """Advance simulation by one timestep (15 minutes)."""
        if self.state.blackout:
            return
        self.state.timestep += 1
        self.state.elapsed_minutes += self.dt_min

        # 1. Fire scheduled events
        self._fire_events()

        # 2. Update weather and renewable output
        self.weather.advance(self.dt_min)
        self._update_weather_and_renewables()

        # 3. Update demand
        self._update_demand()

        # 4. Advance all generators (ramp, startup, trips)
        for gen in self.generators.values():
            gen.advance(self.dt_min, self.rng, self.state.temperature_c)

        # 5. Update battery (just track state, power set by agent)
        # Battery power already applied by control_battery tool

        # 6. Compute aggregates
        self._update_aggregates()

        # 7. Frequency dynamics
        self._update_frequency()

        # 8. UFLS check
        self._check_ufls()

        # 9. Transmission check
        self._check_transmission()

        # 10. Calculate costs
        self._calculate_costs()

        # 11. Blackout check
        if self.state.frequency_hz < BLACKOUT_FREQ_HZ:
            self.state.blackout = True

    # -----------------------------------------------------------------
    # Sub-routines
    # -----------------------------------------------------------------

    def _fire_events(self) -> None:
        for event in self.events:
            if event.fired:
                continue
            if event.timestep != self.state.timestep:
                continue
            event.fired = True
            self._apply_event(event)

    def _apply_event(self, event: ScheduledEvent) -> None:
        if event.event_type == "generator_trip":
            gen = self.generators.get(event.target)
            if gen and gen.status == GeneratorStatus.ONLINE:
                gen.trip()

        elif event.event_type == "generator_derate":
            gen = self.generators.get(event.target)
            if gen:
                new_cap = event.params.get("capacity_mw", gen.spec.capacity_mw * 0.5)
                gen.derate(new_cap)

        elif event.event_type == "line_outage":
            for line in self.lines:
                if line.line_id == event.target:
                    line.status = "tripped"
                    line.current_flow_mw = 0.0

        elif event.event_type == "line_restore":
            for line in self.lines:
                if line.line_id == event.target:
                    line.status = "online"

        elif event.event_type == "demand_surge":
            factor = event.params.get("factor", 1.05)
            self.peak_demand_mw *= factor

    def _update_weather_and_renewables(self) -> None:
        ws = self.weather.get_state()
        self.state.temperature_c = ws.temperature_c
        self.state.wind_speed_m_s = ws.wind_speed_m_s
        self.state.cloud_cover_pct = ws.cloud_cover_pct
        self.state.hour_of_day = ws.hour_of_day

        wind_avail = calculate_wind_output(ws, self.wind.nameplate_mw)
        solar_avail = calculate_solar_output(ws, self.solar.nameplate_mw)
        self.wind.update_output(wind_avail)
        self.solar.update_output(solar_avail)

    def _update_demand(self) -> None:
        h = self.state.hour_of_day
        ws = self.weather.get_state()

        # Select load curve
        if self.season in ("summer", "extreme_heat"):
            base = interpolate_load_curve(h, SUMMER_LOAD_CURVE)
        else:
            base = interpolate_load_curve(h, WINTER_LOAD_CURVE)

        # Temperature adjustment
        temp = self.state.temperature_c
        if temp > 25:
            temp_adj = 1.0 + 0.02 * (temp - 25)
        elif temp < 5:
            temp_adj = 1.0 + 0.015 * (5 - temp)
        else:
            temp_adj = 1.0

        # Weekend factor
        weekend_factor = 0.85 if ws.is_weekend else 1.0

        # Small noise
        noise = 1.0 + self.rng.gauss(0, 0.01)

        self.state.total_demand_mw = (
            self.peak_demand_mw * base * temp_adj * weekend_factor * noise
        )
        self.state.total_demand_mw = max(0.0, self.state.total_demand_mw)

    def _update_aggregates(self) -> None:
        # Total conventional generation
        total_gen = sum(
            g.current_output_mw for g in self.generators.values()
            if g.status == GeneratorStatus.ONLINE
        )
        self.state.total_generation_mw = total_gen

        # Renewables
        self.state.wind_output_mw = self.wind.current_output_mw
        self.state.solar_output_mw = self.solar.current_output_mw
        self.state.total_renewable_mw = (self.wind.current_output_mw +
                                         self.solar.current_output_mw)
        self.state.wind_curtailed_mw = self.wind.curtailed_mw
        self.state.solar_curtailed_mw = self.solar.curtailed_mw

        # Battery
        self.state.battery_power_mw = self.battery.current_power_mw
        self.state.battery_soc_pct = self.battery.soc_pct

        # Load shedding
        manual_shed = sum(z.shed_mw for z in self.zones.values())
        self.state.manual_shed_mw = manual_shed
        self.state.total_load_shed_mw = manual_shed + self.state.ufls_shed_mw

        # Spinning reserve
        reserve = sum(
            g.available_ramp_up(self.dt_min)
            for g in self.generators.values()
            if g.status == GeneratorStatus.ONLINE
        )
        reserve += self.battery.available_discharge_mw
        self.state.spinning_reserve_mw = reserve

        # Required reserve: max(largest_unit, 3% load + 3% gen)
        largest = max(
            (g.current_output_mw for g in self.generators.values()
             if g.status == GeneratorStatus.ONLINE),
            default=0.0,
        )
        pct_based = (0.03 * self.state.total_demand_mw +
                     0.03 * self.state.total_generation_mw)
        self.state.required_reserve_mw = max(largest, pct_based)

    def _update_frequency(self) -> None:
        """Quasi-static frequency model with governor droop response."""
        s = self.state

        # Net power: generation + renewables + battery - demand + shed
        net_demand = s.total_demand_mw - s.total_load_shed_mw
        supply = (s.total_generation_mw + s.total_renewable_mw +
                  s.battery_power_mw)
        imbalance_mw = supply - net_demand

        # System inertia H*S (MW*s)
        total_inertia = 0.0
        for gen in self.generators.values():
            if gen.status == GeneratorStatus.ONLINE and gen.current_output_mw > 0:
                h = INERTIA_CONSTANTS.get(gen.spec.unit_type, 3.0)
                total_inertia += h * gen.current_output_mw
        total_inertia = max(total_inertia, 500.0)  # Floor to prevent div-by-zero

        # Frequency change over timestep (swing equation)
        # df/dt = (P_surplus / (2*H_total)) * f_nom
        # We use a shorter effective dt for quasi-static: the governor acts within
        # ~10 seconds, so we model the steady-state after governor response.
        # Effective dt for frequency = min(dt_actual, governor_time)
        dt_s = self.dt_min * 60.0
        effective_dt = min(dt_s, 30.0)  # Governor settles in ~30s

        # Raw frequency change without governor
        df_raw = (imbalance_mw * NOMINAL_FREQ_HZ / (2.0 * total_inertia)) * effective_dt

        # Governor droop response (automatic, proportional to freq deviation)
        freq_after_raw = s.frequency_hz + df_raw
        freq_dev = freq_after_raw - NOMINAL_FREQ_HZ

        governor_mw = 0.0
        if abs(freq_dev) > GOVERNOR_DEADBAND_HZ:
            for gen in self.generators.values():
                if gen.status == GeneratorStatus.ONLINE:
                    # Droop: delta_P = -(delta_f / (R * f_nom)) * P_rated
                    dp = -(freq_dev / (GOVERNOR_DROOP * NOMINAL_FREQ_HZ)) * gen.spec.capacity_mw
                    # Clamp to ramp limits (in 30s effective window)
                    ramp_limit = gen.spec.ramp_rate_mw_per_min * 0.5  # 30 seconds
                    dp = max(-ramp_limit, min(ramp_limit, dp))
                    # Clamp to capacity bounds
                    if dp > 0:
                        dp = min(dp, gen.effective_capacity - gen.current_output_mw)
                    else:
                        dp = max(dp, gen.spec.min_output_mw - gen.current_output_mw)
                    governor_mw += dp

        # Adjusted imbalance after governor
        adj_imbalance = imbalance_mw + governor_mw
        df_adj = (adj_imbalance * NOMINAL_FREQ_HZ / (2.0 * total_inertia)) * effective_dt

        # Apply damping (loads are frequency-sensitive: ~2%/Hz)
        load_damping_factor = 0.02 * net_demand  # MW/Hz
        if load_damping_factor > 0:
            # df_final = df_adj / (1 + D*f_nom/(2*H_total)*dt)
            damping_denom = 1.0 + (load_damping_factor * NOMINAL_FREQ_HZ /
                                   (2.0 * total_inertia)) * effective_dt
            df_adj /= max(damping_denom, 0.1)

        new_freq = s.frequency_hz + df_adj

        # Frequency recovery toward nominal (long-term AGC effect over 15 min)
        recovery_rate = 0.3  # Move 30% of remaining deviation per step
        remaining_dev = new_freq - NOMINAL_FREQ_HZ
        new_freq -= remaining_dev * recovery_rate

        # Physical bounds
        s.frequency_hz = max(57.0, min(61.0, new_freq))

    def _check_ufls(self) -> None:
        """Under-frequency load shedding."""
        s = self.state
        s.ufls_triggered = False
        s.ufls_shed_mw = 0.0

        cumulative_fraction = 0.0
        for freq_threshold, shed_fraction in UFLS_STAGES:
            if s.frequency_hz < freq_threshold:
                cumulative_fraction += shed_fraction
                s.ufls_triggered = True

        if cumulative_fraction > 0:
            s.ufls_shed_mw = s.total_demand_mw * cumulative_fraction
            s.total_load_shed_mw = s.manual_shed_mw + s.ufls_shed_mw

    def _check_transmission(self) -> None:
        """Simplified transmission flow check.

        Uses a DC-power-flow-like approximation: flow is proportional to
        net injection difference between zones.
        """
        # Calculate net injection per zone (generation - demand + shed)
        zone_gen: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
        zone_demand: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}

        for gen in self.generators.values():
            if gen.status == GeneratorStatus.ONLINE:
                zone_gen[gen.spec.zone] += gen.current_output_mw

        # Renewables: wind in zone C, solar split B/C
        zone_gen["C"] += self.wind.current_output_mw
        zone_gen["B"] += self.solar.current_output_mw * 0.5
        zone_gen["C"] += self.solar.current_output_mw * 0.5

        # Battery in zone B
        zone_gen["B"] += max(0, self.battery.current_power_mw)

        net_demand = self.state.total_demand_mw - self.state.total_load_shed_mw
        for z in self.zones.values():
            zone_demand[z.zone_id] = net_demand * z.demand_fraction

        # Net injection
        net_inj = {z: zone_gen[z] - zone_demand[z] for z in zone_gen}

        # Simple flow model: flow on each line proportional to net injection diff
        for line in self.lines:
            if line.status == "tripped":
                line.current_flow_mw = 0.0
                continue
            inj_from = net_inj.get(line.from_zone, 0.0)
            inj_to = net_inj.get(line.to_zone, 0.0)
            # Flow from surplus zone to deficit zone
            flow = (inj_from - inj_to) / 2.0  # Split between paths
            line.current_flow_mw = max(-line.capacity_mw,
                                       min(line.capacity_mw, flow))

    def _calculate_costs(self) -> None:
        dt_hours = self.dt_min / 60.0
        total_cost = 0.0
        for gen in self.generators.values():
            if gen.status == GeneratorStatus.ONLINE and gen.current_output_mw > 0:
                total_cost += gen.generation_cost_per_hour() * dt_hours
        self.state.generation_cost_usd = total_cost
        self.state.cumulative_cost_usd += total_cost

        # LMP approximation: marginal cost of the most expensive running unit
        marginal_costs = []
        for gen in self.generators.values():
            if gen.status == GeneratorStatus.ONLINE and gen.current_output_mw > 0:
                mc = (2 * gen.spec.cost_a * gen.current_output_mw + gen.spec.cost_b)
                marginal_costs.append(mc)
        self.state.lmp_usd_per_mwh = max(marginal_costs) if marginal_costs else 0.0

    # -----------------------------------------------------------------
    # Display formatting
    # -----------------------------------------------------------------

    def format_status(self) -> str:
        """Format complete grid status for display."""
        s = self.state
        hours = int(s.hour_of_day)
        minutes = int((s.hour_of_day % 1) * 60)

        lines = []
        lines.append("=" * 60)
        lines.append("  POWER GRID CONTROL ROOM")
        lines.append(f"  Step: {s.timestep}/{self.max_steps}  "
                     f"Time: {hours:02d}:{minutes:02d}  "
                     f"({s.elapsed_minutes:.0f} min elapsed)")
        lines.append("=" * 60)
        lines.append("")

        # System summary
        lines.append("--- SYSTEM STATUS ---")
        lines.append(f"  Frequency:       {s.frequency_hz:.3f} Hz")
        lines.append(f"  Total Demand:    {s.total_demand_mw:.0f} MW")
        total_supply = s.total_generation_mw + s.total_renewable_mw + s.battery_power_mw
        lines.append(f"  Total Supply:    {total_supply:.0f} MW")
        lines.append(f"  Thermal Gen:     {s.total_generation_mw:.0f} MW")
        lines.append(f"  Renewable:       {s.total_renewable_mw:.0f} MW "
                     f"(Wind: {s.wind_output_mw:.0f}, Solar: {s.solar_output_mw:.0f})")
        bat_status = "Discharging" if s.battery_power_mw > 0 else (
            "Charging" if s.battery_power_mw < 0 else "Idle")
        lines.append(f"  Battery:         {bat_status} {abs(s.battery_power_mw):.0f} MW "
                     f"(SoC: {s.battery_soc_pct:.1f}%)")
        lines.append(f"  Load Shed:       {s.total_load_shed_mw:.0f} MW"
                     + (" [UFLS ACTIVE]" if s.ufls_triggered else ""))
        lines.append(f"  Spinning Reserve: {s.spinning_reserve_mw:.0f} MW "
                     f"(Required: {s.required_reserve_mw:.0f} MW)"
                     + (" [DEFICIT]" if s.spinning_reserve_mw < s.required_reserve_mw else ""))
        lines.append(f"  LMP:             ${s.lmp_usd_per_mwh:.2f}/MWh")
        lines.append(f"  Step Cost:       ${s.generation_cost_usd:,.0f}")
        lines.append(f"  Cumulative Cost: ${s.cumulative_cost_usd:,.0f}")
        lines.append("")

        # Generators
        lines.append("--- GENERATORS ---")
        for uid in GENERATOR_FLEET:
            gen = self.generators[uid]
            d = gen.to_dict()
            status_str = d["status"].upper()
            if gen.status == GeneratorStatus.ONLINE:
                lines.append(
                    f"  {uid:<12s} {status_str:<10s} "
                    f"{d['current_output_mw']:>6.0f}/{d['capacity_mw']:.0f} MW  "
                    f"[${d['cost_per_mwh']:.1f}/MWh]  Zone {d['zone']}"
                )
            elif gen.status == GeneratorStatus.STARTING:
                lines.append(
                    f"  {uid:<12s} {status_str:<10s} "
                    f"({gen.startup_remaining_min:.0f} min remaining)"
                )
            else:
                lines.append(f"  {uid:<12s} {status_str}")
        lines.append("")

        # Weather
        lines.append("--- WEATHER ---")
        lines.append(f"  Temperature:  {s.temperature_c:.1f} C")
        lines.append(f"  Wind Speed:   {s.wind_speed_m_s:.1f} m/s")
        lines.append(f"  Cloud Cover:  {s.cloud_cover_pct:.0f}%")
        lines.append("")

        # Transmission
        lines.append("--- TRANSMISSION ---")
        for line in self.lines:
            d = line.to_dict()
            if line.status == "tripped":
                lines.append(f"  {d['line_id']}: TRIPPED")
            else:
                lines.append(
                    f"  {d['line_id']}: {d['current_flow_mw']:.0f}/{d['capacity_mw']:.0f} MW "
                    f"({d['loading_pct']:.0f}%)"
                )
        lines.append("")

        if s.blackout:
            lines.append("*** BLACKOUT - TOTAL SYSTEM COLLAPSE ***")
        if s.ufls_triggered:
            lines.append(f"*** UFLS ACTIVE - Auto-shed: {s.ufls_shed_mw:.0f} MW ***")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_state_dict(self) -> dict:
        """Return full state as a serializable dictionary."""
        d = {
            "timestep": self.state.timestep,
            "elapsed_minutes": self.state.elapsed_minutes,
            "hour_of_day": round(self.state.hour_of_day, 2),
            "frequency_hz": round(self.state.frequency_hz, 3),
            "total_generation_mw": round(self.state.total_generation_mw, 1),
            "total_demand_mw": round(self.state.total_demand_mw, 1),
            "total_renewable_mw": round(self.state.total_renewable_mw, 1),
            "wind_output_mw": round(self.state.wind_output_mw, 1),
            "solar_output_mw": round(self.state.solar_output_mw, 1),
            "battery_power_mw": round(self.state.battery_power_mw, 1),
            "battery_soc_pct": round(self.state.battery_soc_pct, 1),
            "total_load_shed_mw": round(self.state.total_load_shed_mw, 1),
            "spinning_reserve_mw": round(self.state.spinning_reserve_mw, 1),
            "required_reserve_mw": round(self.state.required_reserve_mw, 1),
            "generation_cost_usd": round(self.state.generation_cost_usd, 2),
            "cumulative_cost_usd": round(self.state.cumulative_cost_usd, 2),
            "lmp_usd_per_mwh": round(self.state.lmp_usd_per_mwh, 2),
            "temperature_c": self.state.temperature_c,
            "wind_speed_m_s": self.state.wind_speed_m_s,
            "cloud_cover_pct": self.state.cloud_cover_pct,
            "wind_curtailed_mw": round(self.state.wind_curtailed_mw, 1),
            "solar_curtailed_mw": round(self.state.solar_curtailed_mw, 1),
            "ufls_triggered": self.state.ufls_triggered,
            "blackout": self.state.blackout,
            "generators": {uid: g.to_dict() for uid, g in self.generators.items()},
            "battery": self.battery.to_dict(),
            "lines": [l.to_dict() for l in self.lines],
            "zones": {z.zone_id: z.to_dict() for z in self.zones.values()},
        }
        return d
