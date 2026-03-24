"""Deterministic weather model for the power grid simulation.

Provides temperature, wind speed, cloud cover, and solar irradiance at each
timestep. All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class WeatherState:
    """Current weather conditions."""
    temperature_c: float
    wind_speed_m_s: float
    cloud_cover_pct: float        # 0-100
    solar_irradiance_w_m2: float  # Effective (after clouds)
    hour_of_day: float            # 0.0 - 24.0
    day_of_year: int              # 1-365
    is_weekend: bool


@dataclass
class WeatherEvent:
    """A scheduled weather disturbance."""
    start_step: int
    duration_steps: int
    event_type: str     # "wind_ramp_down", "cloud_surge", "temp_plunge", "temp_spike"
    params: dict[str, float]


# =============================================================================
# Season configurations
# =============================================================================

SEASON_CONFIG: dict[str, dict[str, Any]] = {
    "summer": {
        "temp_mean": 32.0,
        "temp_amplitude": 8.0,
        "base_wind_m_s": 7.0,
        "wind_sigma": 2.0,
        "base_cloud_pct": 20.0,
        "cloud_sigma": 10.0,
        "sunrise_hour": 6.0,
        "sunset_hour": 20.0,
        "i_max_clear": 1000.0,
        "day_of_year": 200,
        "is_weekend": False,
    },
    "winter": {
        "temp_mean": 2.0,
        "temp_amplitude": 5.0,
        "base_wind_m_s": 9.0,
        "wind_sigma": 3.0,
        "base_cloud_pct": 40.0,
        "cloud_sigma": 15.0,
        "sunrise_hour": 7.5,
        "sunset_hour": 17.0,
        "i_max_clear": 600.0,
        "day_of_year": 15,
        "is_weekend": False,
    },
    "extreme_cold": {
        "temp_mean": -18.0,
        "temp_amplitude": 4.0,
        "base_wind_m_s": 6.0,
        "wind_sigma": 2.5,
        "base_cloud_pct": 60.0,
        "cloud_sigma": 15.0,
        "sunrise_hour": 7.5,
        "sunset_hour": 17.0,
        "i_max_clear": 500.0,
        "day_of_year": 40,
        "is_weekend": False,
    },
    "extreme_heat": {
        "temp_mean": 38.0,
        "temp_amplitude": 6.0,
        "base_wind_m_s": 4.0,
        "wind_sigma": 1.5,
        "base_cloud_pct": 10.0,
        "cloud_sigma": 8.0,
        "sunrise_hour": 5.5,
        "sunset_hour": 20.5,
        "i_max_clear": 1050.0,
        "day_of_year": 210,
        "is_weekend": False,
    },
    "spring_windy": {
        "temp_mean": 18.0,
        "temp_amplitude": 6.0,
        "base_wind_m_s": 12.0,
        "wind_sigma": 3.0,
        "base_cloud_pct": 25.0,
        "cloud_sigma": 12.0,
        "sunrise_hour": 6.5,
        "sunset_hour": 19.5,
        "i_max_clear": 900.0,
        "day_of_year": 100,
        "is_weekend": True,
    },
}


# =============================================================================
# Wind power curve
# =============================================================================

def wind_power_fraction(wind_speed_m_s: float) -> float:
    """Wind turbine power curve (IEC cubic model).

    Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s.
    Returns fraction of nameplate capacity [0, 1].
    """
    v = wind_speed_m_s
    v_ci = 3.0
    v_r = 12.0
    v_co = 25.0
    if v < v_ci or v >= v_co:
        return 0.0
    if v >= v_r:
        return 1.0
    return (v ** 3 - v_ci ** 3) / (v_r ** 3 - v_ci ** 3)


def solar_power_fraction(irradiance_w_m2: float, temperature_c: float) -> float:
    """Solar PV output as fraction of nameplate.

    Includes temperature derating: -0.4%/deg above 25C.
    """
    if irradiance_w_m2 <= 0:
        return 0.0
    base = irradiance_w_m2 / 1000.0
    temp_factor = 1.0 - 0.004 * max(0.0, temperature_c - 25.0)
    return max(0.0, min(1.0, base * temp_factor))


# =============================================================================
# Weather Model
# =============================================================================

class WeatherModel:
    """Deterministic weather model for grid simulation."""

    def __init__(self, season: str, seed: int = 42,
                 start_hour: float = 0.0,
                 events: list[dict] | None = None) -> None:
        if season not in SEASON_CONFIG:
            raise ValueError(f"Unknown season: {season}. Available: {list(SEASON_CONFIG)}")
        self.config = SEASON_CONFIG[season]
        self.rng = random.Random(seed)
        self.current_hour = start_hour
        self.current_step = 0

        # O-U noise states
        self._temp_noise = 0.0
        self._wind_noise = 0.0
        self._cloud_noise = 0.0
        self._wind_speed = self.config["base_wind_m_s"]

        # Parse events
        self.events: list[WeatherEvent] = []
        if events:
            for e in events:
                self.events.append(WeatherEvent(
                    start_step=e["start_step"],
                    duration_steps=e["duration_steps"],
                    event_type=e["event_type"],
                    params=e.get("params", {}),
                ))

        # Compute initial state
        self._state = self._compute_state()

    def get_state(self) -> WeatherState:
        return self._state

    def advance(self, dt_min: float) -> WeatherState:
        """Advance time and return new weather state."""
        self.current_hour += dt_min / 60.0
        if self.current_hour >= 24.0:
            self.current_hour -= 24.0
        self.current_step += 1
        self._update_noise(dt_min)
        self._state = self._compute_state()
        return self._state

    def _update_noise(self, dt_min: float) -> None:
        """Update Ornstein-Uhlenbeck noise processes."""
        dt_hr = dt_min / 60.0

        # Temperature noise (tau = 3 hours)
        tau_temp = 3.0
        alpha_t = math.exp(-dt_hr / tau_temp)
        self._temp_noise = (alpha_t * self._temp_noise +
                            math.sqrt(1 - alpha_t ** 2) * self.rng.gauss(0, 1.5))

        # Wind noise (tau = 2 hours, autocorrelated)
        tau_wind = 2.0
        alpha_w = math.exp(-dt_hr / tau_wind)
        self._wind_noise = (alpha_w * self._wind_noise +
                            math.sqrt(1 - alpha_w ** 2) *
                            self.rng.gauss(0, self.config["wind_sigma"]))

        # Cloud noise (tau = 1.5 hours)
        tau_cloud = 1.5
        alpha_c = math.exp(-dt_hr / tau_cloud)
        self._cloud_noise = (alpha_c * self._cloud_noise +
                             math.sqrt(1 - alpha_c ** 2) *
                             self.rng.gauss(0, self.config["cloud_sigma"]))

    def _compute_state(self) -> WeatherState:
        h = self.current_hour
        cfg = self.config

        # Temperature: diurnal cycle + noise + event adjustments
        # Peak at 15:00 (3 PM), trough at 03:00 (3 AM)
        # cos(2π(h-15)/24) = 1 at h=15, -1 at h=3
        temp = (cfg["temp_mean"] +
                cfg["temp_amplitude"] * math.cos(2 * math.pi * (h - 15.0) / 24.0) +
                self._temp_noise)

        # Wind speed: base + noise + event adjustments
        wind = max(0.0, cfg["base_wind_m_s"] + self._wind_noise)

        # Cloud cover: base + noise
        cloud = max(0.0, min(100.0, cfg["base_cloud_pct"] + self._cloud_noise))

        # Apply weather events
        for event in self.events:
            if event.start_step <= self.current_step < event.start_step + event.duration_steps:
                progress = ((self.current_step - event.start_step) /
                            max(event.duration_steps, 1))
                temp, wind, cloud = self._apply_event(
                    event, progress, temp, wind, cloud)

        # Solar irradiance
        sunrise = cfg["sunrise_hour"]
        sunset = cfg["sunset_hour"]
        if sunrise < h < sunset:
            day_progress = (h - sunrise) / (sunset - sunrise)
            clear_sky = cfg["i_max_clear"] * math.sin(math.pi * day_progress)
        else:
            clear_sky = 0.0
        irradiance = clear_sky * (1.0 - 0.75 * cloud / 100.0)
        irradiance = max(0.0, irradiance)

        return WeatherState(
            temperature_c=round(temp, 1),
            wind_speed_m_s=round(wind, 1),
            cloud_cover_pct=round(cloud, 1),
            solar_irradiance_w_m2=round(irradiance, 1),
            hour_of_day=round(h, 2),
            day_of_year=cfg["day_of_year"],
            is_weekend=cfg.get("is_weekend", False),
        )

    def _apply_event(self, event: WeatherEvent, progress: float,
                     temp: float, wind: float, cloud: float
                     ) -> tuple[float, float, float]:
        """Modify weather based on active event."""
        p = event.params
        if event.event_type == "wind_ramp_down":
            from_pct = p.get("from_pct", 80.0)
            to_pct = p.get("to_pct", 5.0)
            target_wind = self.config["base_wind_m_s"] * (
                (from_pct + (to_pct - from_pct) * progress) / from_pct
            )
            wind = max(0.0, target_wind)

        elif event.event_type == "wind_ramp_up":
            from_pct = p.get("from_pct", 30.0)
            to_pct = p.get("to_pct", 95.0)
            target_wind = self.config["base_wind_m_s"] * (
                (from_pct + (to_pct - from_pct) * progress) / from_pct
            )
            wind = max(0.0, min(25.0, target_wind))

        elif event.event_type == "cloud_surge":
            target_cloud = p.get("target_pct", 90.0)
            cloud = cloud + (target_cloud - cloud) * progress

        elif event.event_type == "temp_plunge":
            target_temp = p.get("target_c", -25.0)
            temp = temp + (target_temp - temp) * progress

        elif event.event_type == "temp_spike":
            target_temp = p.get("target_c", 42.0)
            temp = temp + (target_temp - temp) * progress

        return temp, wind, cloud


# =============================================================================
# Renewable output calculator
# =============================================================================

def calculate_wind_output(weather: WeatherState, nameplate_mw: float) -> float:
    """Calculate wind farm output from weather."""
    frac = wind_power_fraction(weather.wind_speed_m_s)
    return nameplate_mw * frac


def calculate_solar_output(weather: WeatherState, nameplate_mw: float) -> float:
    """Calculate solar farm output from weather."""
    frac = solar_power_fraction(weather.solar_irradiance_w_m2,
                                weather.temperature_c)
    return nameplate_mw * frac
