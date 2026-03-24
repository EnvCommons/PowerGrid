"""Generator fleet models, battery storage, and renewable sources.

Defines the stateful simulation of 8 thermal generators, 1 grid-scale
battery, and wind/solar renewable sources for a ~5,000 MW power system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# Enums and Specs
# =============================================================================

class GeneratorStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"
    TRIPPED = "tripped"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GeneratorSpec:
    """Immutable specification for a generator unit."""
    unit_id: str
    unit_type: str              # "nuclear", "coal", "ccgt", "peaker"
    capacity_mw: float
    min_output_mw: float
    ramp_rate_mw_per_min: float
    # Quadratic cost curve: cost_per_hour = a*P^2 + b*P + c  (P in MW)
    cost_a: float               # $/MW^2-h
    cost_b: float               # $/MWh
    cost_c: float               # $/h (no-load cost)
    cold_start_time_min: float
    warm_start_time_min: float  # < 8h since shutdown
    hot_start_time_min: float   # < 2h since shutdown
    start_cost_usd: float       # Cold start cost
    min_up_time_min: float
    min_down_time_min: float
    forced_outage_rate: float   # Base probability of trip per 15-min step
    heat_rate_btu_per_kwh: float
    zone: str = "A"             # Transmission zone


# =============================================================================
# Generator Fleet (realistic US-based medium utility)
# =============================================================================

GENERATOR_FLEET: dict[str, GeneratorSpec] = {
    # Forced outage rates are per-15-minute-step probabilities.
    # Derived from NERC GADS annual FOR data (nuclear ~2%, coal ~6%, ccgt ~4%,
    # peaker ~8%) divided by ~35,000 15-min periods/year, then scaled up ~10x
    # for interesting RL dynamics while keeping multi-trip events rare.
    "nuclear_1": GeneratorSpec(
        unit_id="nuclear_1", unit_type="nuclear",
        capacity_mw=1200.0, min_output_mw=600.0,
        ramp_rate_mw_per_min=18.0,
        cost_a=0.00008, cost_b=8.0, cost_c=500.0,
        cold_start_time_min=2880.0, warm_start_time_min=1440.0,
        hot_start_time_min=720.0, start_cost_usd=500000.0,
        min_up_time_min=1440.0, min_down_time_min=1440.0,
        forced_outage_rate=0.00005, heat_rate_btu_per_kwh=10400,
        zone="A",
    ),
    "coal_1": GeneratorSpec(
        unit_id="coal_1", unit_type="coal",
        capacity_mw=500.0, min_output_mw=200.0,
        ramp_rate_mw_per_min=10.0,
        cost_a=0.0003, cost_b=25.0, cost_c=300.0,
        cold_start_time_min=720.0, warm_start_time_min=360.0,
        hot_start_time_min=120.0, start_cost_usd=50000.0,
        min_up_time_min=480.0, min_down_time_min=480.0,
        forced_outage_rate=0.0003, heat_rate_btu_per_kwh=9800,
        zone="A",
    ),
    "coal_2": GeneratorSpec(
        unit_id="coal_2", unit_type="coal",
        capacity_mw=400.0, min_output_mw=160.0,
        ramp_rate_mw_per_min=8.0,
        cost_a=0.0004, cost_b=28.0, cost_c=250.0,
        cold_start_time_min=720.0, warm_start_time_min=360.0,
        hot_start_time_min=120.0, start_cost_usd=45000.0,
        min_up_time_min=480.0, min_down_time_min=480.0,
        forced_outage_rate=0.0004, heat_rate_btu_per_kwh=10200,
        zone="A",
    ),
    "ccgt_1": GeneratorSpec(
        unit_id="ccgt_1", unit_type="ccgt",
        capacity_mw=500.0, min_output_mw=225.0,
        ramp_rate_mw_per_min=30.0,
        cost_a=0.0002, cost_b=35.0, cost_c=200.0,
        cold_start_time_min=240.0, warm_start_time_min=120.0,
        hot_start_time_min=30.0, start_cost_usd=25000.0,
        min_up_time_min=240.0, min_down_time_min=120.0,
        forced_outage_rate=0.0005, heat_rate_btu_per_kwh=6900,
        zone="B",
    ),
    "ccgt_2": GeneratorSpec(
        unit_id="ccgt_2", unit_type="ccgt",
        capacity_mw=400.0, min_output_mw=180.0,
        ramp_rate_mw_per_min=24.0,
        cost_a=0.00025, cost_b=38.0, cost_c=180.0,
        cold_start_time_min=240.0, warm_start_time_min=120.0,
        hot_start_time_min=30.0, start_cost_usd=22000.0,
        min_up_time_min=240.0, min_down_time_min=120.0,
        forced_outage_rate=0.0005, heat_rate_btu_per_kwh=7100,
        zone="B",
    ),
    "peaker_1": GeneratorSpec(
        unit_id="peaker_1", unit_type="peaker",
        capacity_mw=200.0, min_output_mw=50.0,
        ramp_rate_mw_per_min=30.0,
        cost_a=0.0005, cost_b=55.0, cost_c=100.0,
        cold_start_time_min=20.0, warm_start_time_min=10.0,
        hot_start_time_min=5.0, start_cost_usd=5000.0,
        min_up_time_min=30.0, min_down_time_min=15.0,
        forced_outage_rate=0.0008, heat_rate_btu_per_kwh=10500,
        zone="B",
    ),
    "peaker_2": GeneratorSpec(
        unit_id="peaker_2", unit_type="peaker",
        capacity_mw=150.0, min_output_mw=37.0,
        ramp_rate_mw_per_min=22.5,
        cost_a=0.0006, cost_b=60.0, cost_c=80.0,
        cold_start_time_min=20.0, warm_start_time_min=10.0,
        hot_start_time_min=5.0, start_cost_usd=4000.0,
        min_up_time_min=30.0, min_down_time_min=15.0,
        forced_outage_rate=0.0008, heat_rate_btu_per_kwh=11000,
        zone="C",
    ),
    "peaker_3": GeneratorSpec(
        unit_id="peaker_3", unit_type="peaker",
        capacity_mw=100.0, min_output_mw=25.0,
        ramp_rate_mw_per_min=15.0,
        cost_a=0.0008, cost_b=65.0, cost_c=60.0,
        cold_start_time_min=20.0, warm_start_time_min=10.0,
        hot_start_time_min=5.0, start_cost_usd=3000.0,
        min_up_time_min=30.0, min_down_time_min=15.0,
        forced_outage_rate=0.001, heat_rate_btu_per_kwh=11500,
        zone="C",
    ),
}

# Inertia constants by generator type (seconds)
# Sources: Kraljic (2022) arXiv:2210.03661, Kundur "Power System Stability
# and Control" Table 3.1, ERCOT (2018) "Inertia: Basic Concepts".
# Typical ranges: nuclear 3-7s, coal 2.5-6s, ccgt 3-6s, peaker 2-5s.
INERTIA_CONSTANTS: dict[str, float] = {
    "nuclear": 6.0,   # High end; large PWR units (EPRI TR-1000627)
    "coal": 4.0,      # Mid-range (Kraljic 2022 mean ~3.5s)
    "ccgt": 5.0,      # Mid-range (IEC TS 62786)
    "peaker": 3.0,    # Simple-cycle gas turbine
}


# =============================================================================
# Generator Unit (stateful)
# =============================================================================

class GeneratorUnit:
    """A single thermal generator with full operational state machine."""

    def __init__(self, spec: GeneratorSpec, status: str = "offline",
                 output_mw: float = 0.0, derated_capacity_mw: Optional[float] = None) -> None:
        self.spec = spec
        self.status = GeneratorStatus(status)
        self.current_output_mw = output_mw
        self.target_output_mw = output_mw
        self.derated_capacity_mw = derated_capacity_mw
        self.startup_remaining_min: float = 0.0
        self.time_in_state_min: float = 0.0
        self.time_since_shutdown_min: float = 1e6  # Large default = cold

    @property
    def effective_capacity(self) -> float:
        if self.derated_capacity_mw is not None:
            return min(self.spec.capacity_mw, self.derated_capacity_mw)
        return self.spec.capacity_mw

    def set_target(self, target_mw: float) -> tuple[bool, str]:
        """Set generation target. Returns (success, message)."""
        if math.isnan(target_mw) or math.isinf(target_mw):
            return False, f"{self.spec.unit_id}: invalid target value."
        if self.status != GeneratorStatus.ONLINE:
            return False, f"{self.spec.unit_id} is {self.status.value}, cannot dispatch."
        cap = self.effective_capacity
        if target_mw < self.spec.min_output_mw:
            return False, (f"{self.spec.unit_id}: target {target_mw:.1f} MW below minimum "
                           f"{self.spec.min_output_mw:.1f} MW. Use stop_generator to shut down.")
        if target_mw > cap:
            target_mw = cap
        self.target_output_mw = target_mw
        return True, f"{self.spec.unit_id}: target set to {target_mw:.1f} MW."

    def begin_startup(self) -> tuple[bool, str]:
        """Start the unit's startup sequence."""
        if self.status == GeneratorStatus.ONLINE:
            return False, f"{self.spec.unit_id} is already online."
        if self.status == GeneratorStatus.STARTING:
            return False, f"{self.spec.unit_id} is already starting."
        if self.status == GeneratorStatus.UNAVAILABLE:
            return False, f"{self.spec.unit_id} is unavailable."
        if self.status in (GeneratorStatus.OFFLINE, GeneratorStatus.TRIPPED):
            if self.time_since_shutdown_min < self.spec.min_down_time_min:
                remaining = self.spec.min_down_time_min - self.time_since_shutdown_min
                return False, (f"{self.spec.unit_id}: min down time not met. "
                               f"{remaining:.0f} min remaining.")
        # Determine start time based on temperature
        if self.time_since_shutdown_min < 120:  # hot
            start_time = self.spec.hot_start_time_min
        elif self.time_since_shutdown_min < 480:  # warm
            start_time = self.spec.warm_start_time_min
        else:  # cold
            start_time = self.spec.cold_start_time_min
        self.status = GeneratorStatus.STARTING
        self.startup_remaining_min = start_time
        self.time_in_state_min = 0.0
        return True, (f"{self.spec.unit_id}: startup initiated. "
                      f"Estimated time: {start_time:.0f} min.")

    def begin_shutdown(self) -> tuple[bool, str]:
        """Begin orderly shutdown."""
        if self.status != GeneratorStatus.ONLINE:
            return False, f"{self.spec.unit_id} is not online, cannot shut down."
        if self.time_in_state_min < self.spec.min_up_time_min:
            remaining = self.spec.min_up_time_min - self.time_in_state_min
            return False, (f"{self.spec.unit_id}: min up time not met. "
                           f"{remaining:.0f} min remaining.")
        self.status = GeneratorStatus.STOPPING
        self.target_output_mw = 0.0
        self.time_in_state_min = 0.0
        return True, f"{self.spec.unit_id}: shutdown initiated. Ramping to zero."

    def trip(self) -> None:
        """Force immediate offline (equipment failure)."""
        self.status = GeneratorStatus.TRIPPED
        self.current_output_mw = 0.0
        self.target_output_mw = 0.0
        self.time_in_state_min = 0.0
        self.time_since_shutdown_min = 0.0

    def derate(self, new_max_mw: float) -> None:
        """Reduce capacity (e.g. fuel supply curtailment)."""
        self.derated_capacity_mw = new_max_mw
        if self.current_output_mw > new_max_mw:
            self.target_output_mw = new_max_mw

    def advance(self, dt_min: float, rng: random.Random,
                ambient_temp_c: float = 20.0) -> None:
        """Advance generator state by dt_min minutes."""
        self.time_in_state_min += dt_min

        if self.status == GeneratorStatus.STARTING:
            self.startup_remaining_min -= dt_min
            if self.startup_remaining_min <= 0:
                self.status = GeneratorStatus.ONLINE
                self.current_output_mw = self.spec.min_output_mw
                self.target_output_mw = self.spec.min_output_mw
                self.time_in_state_min = 0.0
            return

        if self.status == GeneratorStatus.STOPPING:
            # Ramp down
            max_change = self.spec.ramp_rate_mw_per_min * dt_min
            self.current_output_mw = max(0.0, self.current_output_mw - max_change)
            if self.current_output_mw <= 0:
                self.status = GeneratorStatus.OFFLINE
                self.current_output_mw = 0.0
                self.time_in_state_min = 0.0
                self.time_since_shutdown_min = 0.0
            return

        if self.status in (GeneratorStatus.OFFLINE, GeneratorStatus.TRIPPED,
                           GeneratorStatus.UNAVAILABLE):
            self.time_since_shutdown_min += dt_min
            return

        if self.status == GeneratorStatus.ONLINE:
            # Check for forced outage (trip)
            stress = 1.0
            if self.current_output_mw > 0.95 * self.effective_capacity:
                stress = 2.0
            if ambient_temp_c < -15 or ambient_temp_c > 42:
                stress *= 1.5
            if ambient_temp_c < -25:
                stress *= 2.0
            trip_prob = self.spec.forced_outage_rate * stress
            if rng.random() < trip_prob:
                self.trip()
                return

            # Ramp toward target
            max_change = self.spec.ramp_rate_mw_per_min * dt_min
            delta = self.target_output_mw - self.current_output_mw
            actual_change = max(-max_change, min(max_change, delta))
            self.current_output_mw += actual_change
            # Enforce hard bounds
            cap = self.effective_capacity
            self.current_output_mw = max(
                self.spec.min_output_mw,
                min(cap, self.current_output_mw)
            )

    def generation_cost_per_hour(self) -> float:
        """Quadratic fuel cost: a*P^2 + b*P + c  ($/hr)."""
        p = self.current_output_mw
        if p <= 0:
            return 0.0
        return self.spec.cost_a * p * p + self.spec.cost_b * p + self.spec.cost_c

    def available_ramp_up(self, dt_min: float) -> float:
        """MW headroom available this timestep."""
        if self.status != GeneratorStatus.ONLINE:
            return 0.0
        max_change = self.spec.ramp_rate_mw_per_min * dt_min
        headroom = self.effective_capacity - self.current_output_mw
        return min(max_change, headroom)

    def available_ramp_down(self, dt_min: float) -> float:
        """MW reduction available this timestep."""
        if self.status != GeneratorStatus.ONLINE:
            return 0.0
        max_change = self.spec.ramp_rate_mw_per_min * dt_min
        footroom = self.current_output_mw - self.spec.min_output_mw
        return min(max_change, footroom)

    def to_dict(self) -> dict:
        return {
            "unit_id": self.spec.unit_id,
            "unit_type": self.spec.unit_type,
            "status": self.status.value,
            "current_output_mw": round(self.current_output_mw, 1),
            "target_output_mw": round(self.target_output_mw, 1),
            "capacity_mw": round(self.effective_capacity, 1),
            "min_output_mw": self.spec.min_output_mw,
            "zone": self.spec.zone,
            "cost_per_mwh": round(
                (self.spec.cost_a * self.current_output_mw + self.spec.cost_b +
                 self.spec.cost_c / max(self.current_output_mw, 1.0)), 2
            ) if self.current_output_mw > 0 else 0.0,
        }


# =============================================================================
# Battery Storage
# =============================================================================

class BatteryStorage:
    """Grid-scale lithium-ion battery energy storage system."""

    def __init__(self, capacity_mwh: float = 800.0, max_power_mw: float = 200.0,
                 initial_soc_mwh: float = 400.0,
                 round_trip_efficiency: float = 0.85,
                 soc_min_pct: float = 10.0, soc_max_pct: float = 90.0) -> None:
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.soc_mwh = initial_soc_mwh
        self.round_trip_efficiency = round_trip_efficiency
        self.soc_min_mwh = capacity_mwh * soc_min_pct / 100.0
        self.soc_max_mwh = capacity_mwh * soc_max_pct / 100.0
        self.status: str = "idle"  # "charging", "discharging", "idle"
        self.current_power_mw: float = 0.0  # positive = discharge, negative = charge

    @property
    def soc_pct(self) -> float:
        return (self.soc_mwh / self.capacity_mwh) * 100.0

    @property
    def available_discharge_mw(self) -> float:
        """Max discharge power given current SoC."""
        energy_above_min = self.soc_mwh - self.soc_min_mwh
        if energy_above_min <= 0:
            return 0.0
        return self.max_power_mw

    @property
    def available_charge_mw(self) -> float:
        """Max charge power given current SoC."""
        energy_below_max = self.soc_max_mwh - self.soc_mwh
        if energy_below_max <= 0:
            return 0.0
        return self.max_power_mw

    def charge(self, power_mw: float, dt_hours: float) -> float:
        """Charge battery. Returns actual power consumed from grid (MW)."""
        power_mw = min(power_mw, self.max_power_mw)
        energy_space = self.soc_max_mwh - self.soc_mwh
        # Efficiency applied on charge side
        charge_eff = math.sqrt(self.round_trip_efficiency)
        max_energy_in = energy_space / charge_eff
        max_power_for_soc = max_energy_in / max(dt_hours, 1e-6)
        actual_power = min(power_mw, max_power_for_soc)
        actual_power = max(0.0, actual_power)
        energy_stored = actual_power * dt_hours * charge_eff
        self.soc_mwh = min(self.soc_max_mwh, self.soc_mwh + energy_stored)
        self.current_power_mw = -actual_power
        self.status = "charging" if actual_power > 0 else "idle"
        return actual_power

    def discharge(self, power_mw: float, dt_hours: float) -> float:
        """Discharge battery. Returns actual power delivered to grid (MW)."""
        power_mw = min(power_mw, self.max_power_mw)
        energy_available = self.soc_mwh - self.soc_min_mwh
        discharge_eff = math.sqrt(self.round_trip_efficiency)
        max_energy_out = energy_available * discharge_eff
        max_power_for_soc = max_energy_out / max(dt_hours, 1e-6)
        actual_power = min(power_mw, max_power_for_soc)
        actual_power = max(0.0, actual_power)
        energy_withdrawn = actual_power * dt_hours / discharge_eff
        self.soc_mwh = max(self.soc_min_mwh, self.soc_mwh - energy_withdrawn)
        self.current_power_mw = actual_power
        self.status = "discharging" if actual_power > 0 else "idle"
        return actual_power

    def set_idle(self) -> None:
        self.current_power_mw = 0.0
        self.status = "idle"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "current_power_mw": round(self.current_power_mw, 1),
            "soc_pct": round(self.soc_pct, 1),
            "soc_mwh": round(self.soc_mwh, 1),
            "capacity_mwh": self.capacity_mwh,
            "max_power_mw": self.max_power_mw,
            "available_discharge_mw": round(self.available_discharge_mw, 1),
            "available_charge_mw": round(self.available_charge_mw, 1),
        }


# =============================================================================
# Renewable Source
# =============================================================================

class RenewableSource:
    """Wind or solar renewable generation source."""

    def __init__(self, source_type: str, nameplate_mw: float) -> None:
        self.source_type = source_type  # "wind" or "solar"
        self.nameplate_mw = nameplate_mw
        self.current_output_mw: float = 0.0
        self.available_output_mw: float = 0.0  # Before curtailment
        self.curtailment_limit_mw: Optional[float] = None

    def update_output(self, available_mw: float) -> None:
        """Set available output from weather model, apply curtailment."""
        self.available_output_mw = min(available_mw, self.nameplate_mw)
        if self.curtailment_limit_mw is not None:
            self.current_output_mw = min(self.available_output_mw,
                                         self.curtailment_limit_mw)
        else:
            self.current_output_mw = self.available_output_mw

    def curtail(self, limit_mw: float) -> str:
        """Set curtailment limit."""
        self.curtailment_limit_mw = max(0.0, limit_mw)
        self.current_output_mw = min(self.available_output_mw,
                                     self.curtailment_limit_mw)
        return (f"{self.source_type}: curtailed to {limit_mw:.1f} MW "
                f"(available: {self.available_output_mw:.1f} MW)")

    def uncurtail(self) -> str:
        """Remove curtailment."""
        self.curtailment_limit_mw = None
        self.current_output_mw = self.available_output_mw
        return f"{self.source_type}: curtailment removed."

    @property
    def curtailed_mw(self) -> float:
        return max(0.0, self.available_output_mw - self.current_output_mw)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "nameplate_mw": self.nameplate_mw,
            "current_output_mw": round(self.current_output_mw, 1),
            "available_output_mw": round(self.available_output_mw, 1),
            "curtailed_mw": round(self.curtailed_mw, 1),
        }
