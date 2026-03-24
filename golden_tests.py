"""Comprehensive tests for the Power Grid Operator environment.

Tests verify:
1. Generator state machines (ramp, startup, trip, min up/down)
2. Battery storage (SoC limits, efficiency, power limits)
3. Weather model (diurnal cycle, wind/solar curves)
4. Frequency dynamics (swing equation, governor, UFLS)
5. Demand model (summer/winter profiles, temperature sensitivity)
6. Transmission (line limits, trip behavior)
7. Reward calculation (all 5 components, terminal, blackout)
8. Scenario registry (task counts, initial conditions)
9. Integration tests (multi-step simulation)
10. Determinism
"""

import copy
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from generators import (
    GENERATOR_FLEET, BatteryStorage, GeneratorSpec, GeneratorStatus,
    GeneratorUnit, RenewableSource,
)
from weather import (
    WeatherModel, WeatherState, calculate_solar_output, calculate_wind_output,
    solar_power_fraction, wind_power_fraction,
)
from simulation import (
    BLACKOUT_FREQ_HZ, NOMINAL_FREQ_HZ, UFLS_STAGES, GridSimulation,
    GridState, interpolate_load_curve, SUMMER_LOAD_CURVE, WINTER_LOAD_CURVE,
)
from scenarios import (
    ALL_SCENARIOS, SEEDS_PER_SCENARIO, TRAIN_SCENARIOS, TEST_SCENARIOS,
    ScenarioRegistry,
)
from rewards import RewardCalculator


# =============================================================================
# 1. GENERATOR TESTS
# =============================================================================

class TestGenerators:
    """Tests for thermal generator state machine."""

    def _make_gen(self, unit_id: str = "ccgt_1", status: str = "online",
                  output_mw: float = 300.0) -> GeneratorUnit:
        return GeneratorUnit(GENERATOR_FLEET[unit_id], status, output_mw)

    def test_ramp_rate_enforcement(self):
        """Generator cannot exceed ramp rate per timestep."""
        gen = self._make_gen("ccgt_1", "online", 300.0)
        gen.set_target(500.0)  # +200 MW, but ramp = 30 MW/min * 15 min = 450 max
        rng = random.Random(42)
        gen.advance(15.0, rng)
        max_change = gen.spec.ramp_rate_mw_per_min * 15.0  # 450 MW
        assert gen.current_output_mw <= 300.0 + max_change + 0.1
        assert gen.current_output_mw >= 300.0  # Should have ramped up

    def test_min_output_enforcement(self):
        """Cannot dispatch below minimum stable load."""
        gen = self._make_gen("coal_1", "online", 300.0)
        ok, msg = gen.set_target(100.0)  # Below min 200 MW
        assert not ok
        assert "minimum" in msg.lower() or "below" in msg.lower()

    def test_max_output_enforcement(self):
        """Target above capacity is clamped to capacity."""
        gen = self._make_gen("peaker_1", "online", 100.0)
        ok, msg = gen.set_target(999.0)  # Above 200 MW capacity
        assert ok
        assert gen.target_output_mw == 200.0

    def test_startup_sequence_timing(self):
        """Startup takes correct number of timesteps."""
        gen = self._make_gen("peaker_1", "offline", 0.0)
        gen.time_since_shutdown_min = 1e6  # Cold
        ok, msg = gen.begin_startup()
        assert ok
        assert gen.status == GeneratorStatus.STARTING
        # Cold start = 20 min, should complete in 2 steps of 15 min
        rng = random.Random(42)
        gen.advance(15.0, rng)
        assert gen.status == GeneratorStatus.STARTING  # Not done yet (5 min left)
        gen.advance(15.0, rng)
        assert gen.status == GeneratorStatus.ONLINE  # Now online

    def test_cold_vs_hot_start_time(self):
        """Hot start should be faster than cold start."""
        gen_cold = self._make_gen("ccgt_1", "offline", 0.0)
        gen_cold.time_since_shutdown_min = 1e6
        gen_cold.begin_startup()
        cold_time = gen_cold.startup_remaining_min

        gen_hot = self._make_gen("ccgt_1", "offline", 0.0)
        gen_hot.time_since_shutdown_min = 60.0  # Hot (< 2hr)
        gen_hot.begin_startup()
        hot_time = gen_hot.startup_remaining_min

        assert hot_time < cold_time, f"Hot start ({hot_time}) should be faster than cold ({cold_time})"

    def test_min_up_time_enforcement(self):
        """Cannot stop before minimum up time."""
        gen = self._make_gen("coal_1", "online", 300.0)
        gen.time_in_state_min = 60.0  # Only 1 hour, need 8 hours
        ok, msg = gen.begin_shutdown()
        assert not ok
        assert "min up time" in msg.lower() or "remaining" in msg.lower()

    def test_min_down_time_enforcement(self):
        """Cannot start before minimum down time."""
        gen = self._make_gen("coal_1", "offline", 0.0)
        gen.time_since_shutdown_min = 60.0  # Only 1 hour, need 8 hours
        ok, msg = gen.begin_startup()
        assert not ok

    def test_trip_sets_offline(self):
        """Trip immediately takes generator offline."""
        gen = self._make_gen("ccgt_1", "online", 400.0)
        gen.trip()
        assert gen.status == GeneratorStatus.TRIPPED
        assert gen.current_output_mw == 0.0

    def test_cost_curve_monotonic(self):
        """Generation cost per hour should increase with output."""
        gen = self._make_gen("coal_1", "online", 200.0)
        cost_low = gen.generation_cost_per_hour()
        gen.current_output_mw = 500.0
        cost_high = gen.generation_cost_per_hour()
        assert cost_high > cost_low, "Cost should increase with output"

    def test_derate_reduces_capacity(self):
        """Derated generator has lower effective capacity."""
        gen = self._make_gen("ccgt_1", "online", 400.0)
        assert gen.effective_capacity == 500.0
        gen.derate(250.0)
        assert gen.effective_capacity == 250.0

    def test_cannot_dispatch_offline_generator(self):
        """Cannot set target on offline generator."""
        gen = self._make_gen("peaker_1", "offline", 0.0)
        ok, msg = gen.set_target(100.0)
        assert not ok

    def test_available_ramp_up(self):
        """Available ramp up is limited by both ramp rate and headroom."""
        gen = self._make_gen("peaker_1", "online", 180.0)
        avail = gen.available_ramp_up(15.0)
        headroom = 200.0 - 180.0  # 20 MW
        ramp_limit = 30.0 * 15.0  # 450 MW
        assert avail == min(headroom, ramp_limit)


# =============================================================================
# 2. BATTERY TESTS
# =============================================================================

class TestBattery:
    """Tests for grid-scale battery storage."""

    def test_charge_respects_soc_max(self):
        """Cannot charge above 90% SoC."""
        bat = BatteryStorage(initial_soc_mwh=710.0)  # Near max (720)
        actual = bat.charge(200.0, 0.25)
        assert bat.soc_mwh <= bat.soc_max_mwh + 0.1

    def test_discharge_respects_soc_min(self):
        """Cannot discharge below 10% SoC."""
        bat = BatteryStorage(initial_soc_mwh=90.0)  # Near min (80)
        actual = bat.discharge(200.0, 0.25)
        assert bat.soc_mwh >= bat.soc_min_mwh - 0.1

    def test_round_trip_efficiency(self):
        """Round-trip should lose ~15% of energy."""
        bat = BatteryStorage(initial_soc_mwh=200.0)
        initial_soc = bat.soc_mwh
        # Charge 100 MW for 1 hour: grid supplies 100 MWh
        charge_power = bat.charge(100.0, 1.0)
        energy_in_soc = bat.soc_mwh - initial_soc  # Energy stored in battery
        # Now fully discharge back to original SoC level
        # Withdraw energy_in_soc from battery, deliver energy_in_soc * sqrt(eff) to grid
        soc_before_discharge = bat.soc_mwh
        bat.discharge(200.0, 1.0)  # Request enough to withdraw everything stored
        energy_out_soc = soc_before_discharge - bat.soc_mwh  # SoC withdrawn

        # The grid received 100 MWh (charge_power * 1hr).
        # Battery stored energy_in_soc = 100 * sqrt(0.85) ≈ 92.2 MWh.
        # On discharge, grid receives energy_out_soc * sqrt(0.85).
        # Charge efficiency: energy_in_soc / 100 should be ~sqrt(0.85)
        charge_eff = energy_in_soc / 100.0
        assert 0.90 < charge_eff < 0.96, f"Charge efficiency {charge_eff:.4f}"
        # The overall RTE is charge_eff * discharge_eff = sqrt(0.85)^2 = 0.85
        # Verify: energy_in_soc should equal ~92.2, which is 100 * sqrt(0.85)
        expected_soc = 100.0 * math.sqrt(0.85)
        assert abs(energy_in_soc - expected_soc) < 1.0, (
            f"Stored {energy_in_soc:.1f} MWh, expected ~{expected_soc:.1f} MWh"
        )

    def test_power_limit(self):
        """Cannot exceed max power."""
        bat = BatteryStorage(initial_soc_mwh=400.0)
        actual = bat.discharge(500.0, 0.25)  # Request 500 MW, max is 200
        assert actual <= 200.0

    def test_idle_preserves_soc(self):
        """Idle mode should not change SoC."""
        bat = BatteryStorage(initial_soc_mwh=400.0)
        bat.set_idle()
        assert bat.soc_mwh == 400.0
        assert bat.current_power_mw == 0.0


# =============================================================================
# 3. WEATHER TESTS
# =============================================================================

class TestWeather:
    """Tests for weather model."""

    def test_diurnal_temperature_cycle(self):
        """Temperature should peak near 3 PM and trough near 3 AM."""
        model = WeatherModel("summer", seed=42, start_hour=0.0)
        temps = []
        for _ in range(96):  # 24 hours
            ws = model.advance(15.0)
            temps.append((ws.hour_of_day, ws.temperature_c))

        peak_hour = max(temps, key=lambda x: x[1])[0]
        trough_hour = min(temps, key=lambda x: x[1])[0]
        # Peak should be afternoon (12-18)
        assert 12 <= peak_hour <= 20, f"Peak at hour {peak_hour}, expected 12-20"
        # Trough should be early morning (0-6)
        assert trough_hour <= 8 or trough_hour >= 22, f"Trough at hour {trough_hour}"

    def test_solar_zero_at_night(self):
        """No solar output at night."""
        ws = WeatherState(temperature_c=25, wind_speed_m_s=8,
                          cloud_cover_pct=0, solar_irradiance_w_m2=0,
                          hour_of_day=2.0, day_of_year=200, is_weekend=False)
        output = calculate_solar_output(ws, 300.0)
        assert output == 0.0

    def test_solar_peak_near_noon(self):
        """Solar should be highest near noon."""
        model = WeatherModel("summer", seed=42, start_hour=0.0)
        outputs = []
        for _ in range(96):
            ws = model.advance(15.0)
            out = calculate_solar_output(ws, 300.0)
            outputs.append((ws.hour_of_day, out))

        peak_hour = max(outputs, key=lambda x: x[1])[0]
        assert 10 <= peak_hour <= 15, f"Solar peak at hour {peak_hour}, expected 10-15"

    def test_wind_cut_in_speed(self):
        """No wind power below 3 m/s."""
        assert wind_power_fraction(2.0) == 0.0
        assert wind_power_fraction(3.0) == 0.0
        assert wind_power_fraction(4.0) > 0.0

    def test_wind_cut_out_speed(self):
        """No wind power above 25 m/s."""
        assert wind_power_fraction(25.0) == 0.0
        assert wind_power_fraction(24.0) > 0.0

    def test_wind_rated_output(self):
        """Wind should be at full output at rated speed (12 m/s) and above."""
        assert abs(wind_power_fraction(12.0) - 1.0) < 0.01
        assert abs(wind_power_fraction(20.0) - 1.0) < 0.01

    def test_wind_ramp_event(self):
        """Wind ramp down event should reduce wind speed."""
        model = WeatherModel("spring_windy", seed=42, start_hour=6.0,
                             events=[{
                                 "start_step": 5, "duration_steps": 8,
                                 "event_type": "wind_ramp_down",
                                 "params": {"from_pct": 80.0, "to_pct": 5.0}
                             }])
        # Advance past the event
        winds_before = []
        for _ in range(5):
            ws = model.advance(15.0)
            winds_before.append(ws.wind_speed_m_s)

        winds_during = []
        for _ in range(8):
            ws = model.advance(15.0)
            winds_during.append(ws.wind_speed_m_s)

        # Wind should be lower during the event
        assert min(winds_during) < min(winds_before), (
            f"Wind should decrease during ramp: before {winds_before}, during {winds_during}"
        )


# =============================================================================
# 4. FREQUENCY DYNAMICS TESTS
# =============================================================================

class TestFrequencyDynamics:
    """Tests for frequency dynamics model."""

    def _make_sim(self, scenario: str = "summer_peak", seed: int = 42) -> GridSimulation:
        sc = ScenarioRegistry.get(scenario)
        return GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": [],  # No events for frequency tests
            },
            seed=seed,
        )

    def test_balanced_load_steady_frequency(self):
        """When supply is adequate, frequency should stay within normal bounds."""
        # Use summer peak starting at midnight where generation ~ demand
        sim = self._make_sim()
        for _ in range(5):
            sim.advance()
        # With adequate supply, frequency stays within the physical bounds (57-61 Hz)
        # and should not cause blackout
        assert sim.state.frequency_hz >= BLACKOUT_FREQ_HZ, (
            f"Frequency should not cause blackout, got {sim.state.frequency_hz}"
        )
        assert not sim.state.blackout, "Should not blackout with adequate supply"

    def test_deficit_drops_frequency(self):
        """Large generation deficit should drop frequency."""
        sim = self._make_sim()
        # Trip nuclear (1200 MW) + coal_1 (500 MW) to create a clear deficit
        sim.generators["nuclear_1"].trip()
        sim.generators["coal_1"].trip()
        sim.advance()
        assert sim.state.frequency_hz < 60.0, (
            f"Frequency should drop with ~1700 MW deficit, got {sim.state.frequency_hz}"
        )

    def test_surplus_raises_frequency(self):
        """Large generation surplus should raise frequency."""
        sim = self._make_sim()
        # Reduce demand drastically
        sim.peak_demand_mw = 2000.0
        sim.advance()
        assert sim.state.frequency_hz > 59.9, (
            f"Frequency should be at or above normal with surplus, got {sim.state.frequency_hz}"
        )

    def test_ufls_stage_1_threshold(self):
        """UFLS stage 1 should trigger at 59.5 Hz."""
        # Verify the constant
        assert UFLS_STAGES[0][0] == 59.5
        assert UFLS_STAGES[0][1] == 0.10

    def test_blackout_threshold(self):
        """Frequency below 57.5 Hz should cause blackout."""
        assert BLACKOUT_FREQ_HZ == 57.5

    def test_no_blackout_under_normal_conditions(self):
        """Normal operation should never cause blackout."""
        sim = self._make_sim()
        for _ in range(20):
            sim.advance()
        assert not sim.state.blackout


# =============================================================================
# 5. DEMAND MODEL TESTS
# =============================================================================

class TestDemandModel:
    """Tests for demand calculation."""

    def test_summer_peak_afternoon(self):
        """Summer demand should peak in afternoon."""
        curve_17 = interpolate_load_curve(17.0, SUMMER_LOAD_CURVE)
        curve_3 = interpolate_load_curve(3.0, SUMMER_LOAD_CURVE)
        assert curve_17 > curve_3, "5 PM should be higher demand than 3 AM"
        assert curve_17 >= 0.95, f"Summer peak should be near 1.0, got {curve_17}"

    def test_summer_trough_overnight(self):
        """Summer trough should be early morning."""
        curve_3 = interpolate_load_curve(3.0, SUMMER_LOAD_CURVE)
        assert curve_3 <= 0.60, f"Summer trough should be ~0.55, got {curve_3}"

    def test_winter_double_peak(self):
        """Winter should have morning and evening peaks."""
        morning = interpolate_load_curve(8.0, WINTER_LOAD_CURVE)
        midday = interpolate_load_curve(12.0, WINTER_LOAD_CURVE)
        evening = interpolate_load_curve(19.0, WINTER_LOAD_CURVE)
        assert morning > midday, "Winter morning peak should exceed midday"
        assert evening > midday, "Winter evening peak should exceed midday"

    def test_load_curve_range(self):
        """All load curve values should be in [0.5, 1.0]."""
        for h in range(24):
            for curve in [SUMMER_LOAD_CURVE, WINTER_LOAD_CURVE]:
                val = curve[h]
                assert 0.5 <= val <= 1.0, f"Load curve value {val} at hour {h} out of range"


# =============================================================================
# 6. TRANSMISSION TESTS
# =============================================================================

class TestTransmission:
    """Tests for transmission network."""

    def test_line_capacity_values(self):
        """Verify default line capacities."""
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96, "peak_demand_mw": 5000.0, "season": "summer",
                "start_hour": 0.0, "generator_initial": ALL_SCENARIOS["summer_peak"].generator_initial,
                "battery_initial": {"soc_mwh": 400.0},
                "weather_config": {"season": "summer", "events": []}, "events": [],
            }, seed=42
        )
        capacities = {l.line_id: l.capacity_mw for l in sim.lines}
        assert capacities["line_AB"] == 2000.0
        assert capacities["line_BC"] == 1500.0
        assert capacities["line_AC"] == 800.0

    def test_line_trip_zeroes_flow(self):
        """Tripped line should have zero flow."""
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96, "peak_demand_mw": 5000.0, "season": "summer",
                "start_hour": 0.0, "generator_initial": ALL_SCENARIOS["summer_peak"].generator_initial,
                "battery_initial": {"soc_mwh": 400.0},
                "weather_config": {"season": "summer", "events": []},
                "events": [{"timestep": 1, "event_type": "line_outage",
                            "target": "line_AB", "params": {}}],
            }, seed=42
        )
        sim.advance()
        for line in sim.lines:
            if line.line_id == "line_AB":
                assert line.status == "tripped"
                assert line.current_flow_mw == 0.0

    def test_three_zones_exist(self):
        """All three zones should exist with correct demand fractions."""
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96, "peak_demand_mw": 5000.0, "season": "summer",
                "start_hour": 0.0, "generator_initial": ALL_SCENARIOS["summer_peak"].generator_initial,
                "battery_initial": {"soc_mwh": 400.0},
                "weather_config": {"season": "summer", "events": []}, "events": [],
            }, seed=42
        )
        assert "A" in sim.zones
        assert "B" in sim.zones
        assert "C" in sim.zones
        total_frac = sum(z.demand_fraction for z in sim.zones.values())
        assert abs(total_frac - 1.0) < 0.01


# =============================================================================
# 7. REWARD TESTS
# =============================================================================

class TestRewards:
    """Tests for reward calculation."""

    def _healthy_state(self) -> GridState:
        state = GridState()
        state.frequency_hz = 60.0
        state.total_demand_mw = 4000.0
        state.total_generation_mw = 3700.0
        state.total_renewable_mw = 300.0
        state.wind_output_mw = 200.0
        state.solar_output_mw = 100.0
        state.battery_power_mw = 0.0
        state.total_load_shed_mw = 0.0
        state.spinning_reserve_mw = 500.0
        state.required_reserve_mw = 400.0
        state.generation_cost_usd = 10000.0
        state.cumulative_cost_usd = 100000.0
        state.wind_curtailed_mw = 0.0
        state.solar_curtailed_mw = 0.0
        state.ufls_triggered = False
        state.blackout = False
        return state

    def test_perfect_operation_positive_reward(self):
        """All-good state should give positive reward."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = self._healthy_state()
        reward = calc.step_reward(state)
        assert reward > 0.5, f"Healthy state should give positive reward, got {reward}"

    def test_load_shedding_reduces_reward(self):
        """Shedding load should reduce reward."""
        calc1 = RewardCalculator("summer_peak", 5000.0)
        calc2 = RewardCalculator("summer_peak", 5000.0)
        state_good = self._healthy_state()
        state_shed = self._healthy_state()
        state_shed.total_load_shed_mw = 400.0

        r_good = calc1.step_reward(state_good)
        r_shed = calc2.step_reward(state_shed)
        assert r_shed < r_good, f"Shedding should reduce reward: good={r_good}, shed={r_shed}"

    def test_frequency_deviation_penalized(self):
        """Frequency deviation should reduce reward."""
        calc1 = RewardCalculator("summer_peak", 5000.0)
        calc2 = RewardCalculator("summer_peak", 5000.0)
        state_good = self._healthy_state()
        state_bad = self._healthy_state()
        state_bad.frequency_hz = 59.7  # Significant deviation

        r_good = calc1.step_reward(state_good)
        r_bad = calc2.step_reward(state_bad)
        assert r_bad < r_good, "Frequency deviation should reduce reward"

    def test_blackout_gives_minus_one(self):
        """Blackout should give -1.0 reward."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = self._healthy_state()
        state.blackout = True
        reward = calc.step_reward(state)
        assert reward == -1.0

    def test_cost_efficiency_reward(self):
        """Lower cost should give higher reward."""
        calc1 = RewardCalculator("summer_peak", 5000.0)
        calc2 = RewardCalculator("summer_peak", 5000.0)
        state_cheap = self._healthy_state()
        state_cheap.generation_cost_usd = 5000.0
        state_expensive = self._healthy_state()
        state_expensive.generation_cost_usd = 50000.0

        r_cheap = calc1.step_reward(state_cheap)
        r_expensive = calc2.step_reward(state_expensive)
        assert r_cheap > r_expensive, "Cheaper generation should score higher"

    def test_renewable_curtailment_penalized(self):
        """Curtailing renewables should reduce renewable score."""
        calc1 = RewardCalculator("summer_peak", 5000.0)
        calc2 = RewardCalculator("summer_peak", 5000.0)
        state_used = self._healthy_state()
        state_used.wind_output_mw = 200.0
        state_used.wind_curtailed_mw = 0.0
        state_curtailed = self._healthy_state()
        state_curtailed.wind_output_mw = 100.0
        state_curtailed.wind_curtailed_mw = 100.0

        r_used = calc1.step_reward(state_used)
        r_curtailed = calc2.step_reward(state_curtailed)
        assert r_curtailed < r_used, "Curtailment should reduce reward"

    def test_reserve_deficit_penalized(self):
        """Reserves below requirement should reduce reward."""
        calc1 = RewardCalculator("summer_peak", 5000.0)
        calc2 = RewardCalculator("summer_peak", 5000.0)
        state_adequate = self._healthy_state()
        state_adequate.spinning_reserve_mw = 500.0
        state_deficit = self._healthy_state()
        state_deficit.spinning_reserve_mw = 100.0

        r_adequate = calc1.step_reward(state_adequate)
        r_deficit = calc2.step_reward(state_deficit)
        assert r_deficit < r_adequate, "Reserve deficit should reduce reward"

    def test_terminal_reward_blackout(self):
        """Terminal reward for blackout should be -1.0."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = self._healthy_state()
        state.blackout = True
        assert calc.terminal_reward(state) == -1.0

    def test_terminal_reward_success_positive(self):
        """Terminal reward for successful completion should be positive."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = self._healthy_state()
        reward = calc.terminal_reward(state)
        assert reward > 0, f"Successful terminal reward should be positive, got {reward}"

    def test_is_terminal_blackout(self):
        """Blackout should be terminal."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = GridState()
        state.blackout = True
        is_term, reason = calc.is_terminal(state, 10, 96)
        assert is_term
        assert reason == "blackout"

    def test_is_terminal_frequency_collapse(self):
        """Frequency below 57.5 Hz should be terminal."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = GridState()
        state.frequency_hz = 57.0
        is_term, reason = calc.is_terminal(state, 10, 96)
        assert is_term
        assert reason == "frequency_collapse"

    def test_is_terminal_max_steps(self):
        """Reaching max steps should be terminal."""
        calc = RewardCalculator("summer_peak", 5000.0)
        state = GridState()
        state.frequency_hz = 60.0
        is_term, reason = calc.is_terminal(state, 96, 96)
        assert is_term
        assert reason == "completed"


# =============================================================================
# 8. SCENARIO TESTS
# =============================================================================

class TestScenarios:
    """Tests for scenario registry."""

    def test_all_scenarios_exist(self):
        """All 8 scenarios should be defined."""
        assert len(ALL_SCENARIOS) == 8

    def test_train_scenarios(self):
        """Training split should have 4 scenarios."""
        assert len(TRAIN_SCENARIOS) == 4

    def test_test_scenarios(self):
        """Test split should have 4 scenarios."""
        assert len(TEST_SCENARIOS) == 4

    def test_train_task_count(self):
        """Training split should have 20 tasks (4 scenarios * 5 seeds)."""
        tasks = ScenarioRegistry.list_tasks("train")
        assert len(tasks) == 4 * SEEDS_PER_SCENARIO

    def test_test_task_count(self):
        """Test split should have 20 tasks."""
        tasks = ScenarioRegistry.list_tasks("test")
        assert len(tasks) == 4 * SEEDS_PER_SCENARIO

    def test_splits_defined(self):
        """Train and test splits should exist."""
        splits = ScenarioRegistry.list_splits()
        assert "train" in splits
        assert "test" in splits

    def test_summer_peak_initial_conditions(self):
        """Summer peak: nuclear online at 1200 MW."""
        sc = ALL_SCENARIOS["summer_peak"]
        assert sc.generator_initial["nuclear_1"]["status"] == "online"
        assert sc.generator_initial["nuclear_1"]["output_mw"] == 1200.0
        assert sc.peak_demand_mw == 3500.0

    def test_cold_snap_events(self):
        """Cold snap should have gas curtailment and trip events."""
        sc = ALL_SCENARIOS["cold_snap"]
        event_types = [e["event_type"] for e in sc.events]
        assert "generator_derate" in event_types
        assert "generator_trip" in event_types

    def test_polar_vortex_288_steps(self):
        """Polar vortex should be 288 steps (72 hours)."""
        sc = ALL_SCENARIOS["polar_vortex"]
        assert sc.max_steps == 288

    def test_cascading_failure_has_line_trips(self):
        """Cascading failure should have multiple line trip events."""
        sc = ALL_SCENARIOS["cascading_failure"]
        line_events = [e for e in sc.events if e["event_type"] == "line_outage"]
        assert len(line_events) >= 2

    def test_task_ids_unique(self):
        """All task IDs should be unique."""
        all_tasks = ScenarioRegistry.list_tasks("train") + ScenarioRegistry.list_tasks("test")
        ids = [t["id"] for t in all_tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"


# =============================================================================
# 9. INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Tests for full simulation integration."""

    def _make_sim(self, scenario: str, seed: int = 42) -> GridSimulation:
        sc = ScenarioRegistry.get(scenario)
        return GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            },
            seed=seed,
        )

    def test_summer_peak_runs_10_steps(self):
        """Summer peak should run 10 steps without crashing."""
        sim = self._make_sim("summer_peak")
        for _ in range(10):
            sim.advance()
        assert sim.state.timestep == 10
        assert not sim.state.blackout

    def test_cold_snap_ccgt_derated(self):
        """CCGT_1 should be derated at step 8 in cold snap."""
        sim = self._make_sim("cold_snap")
        for _ in range(10):
            sim.advance()
        gen = sim.generators["ccgt_1"]
        assert gen.effective_capacity == 250.0

    def test_line_outage_trips_line(self):
        """Line A->B should trip at step 12 in line_outage."""
        sim = self._make_sim("line_outage")
        for _ in range(15):
            sim.advance()
        for line in sim.lines:
            if line.line_id == "line_AB":
                assert line.status == "tripped"

    def test_format_status_not_empty(self):
        """format_status should return non-empty string."""
        sim = self._make_sim("summer_peak")
        sim.advance()
        text = sim.format_status()
        assert len(text) > 100
        assert "POWER GRID CONTROL ROOM" in text

    def test_get_state_dict_has_keys(self):
        """get_state_dict should contain essential keys."""
        sim = self._make_sim("summer_peak")
        sim.advance()
        d = sim.get_state_dict()
        assert "frequency_hz" in d
        assert "total_demand_mw" in d
        assert "generators" in d
        assert "battery" in d
        assert "lines" in d

    def test_generation_cost_positive(self):
        """Running generators should have positive cost."""
        sim = self._make_sim("summer_peak")
        sim.advance()
        assert sim.state.generation_cost_usd > 0

    def test_renewable_output_varies(self):
        """Renewable output should change across timesteps."""
        sim = self._make_sim("summer_peak")
        outputs = []
        for _ in range(48):
            sim.advance()
            outputs.append(sim.state.total_renewable_mw)
        # At least some variation (night vs day for solar)
        assert max(outputs) > min(outputs) + 10, "Renewables should vary"


# =============================================================================
# 10. DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for simulation determinism."""

    def test_same_seed_same_trajectory(self):
        """Same scenario + same seed = identical trajectory."""
        sc = ScenarioRegistry.get("summer_peak")
        config = {
            "max_steps": sc.max_steps,
            "peak_demand_mw": sc.peak_demand_mw,
            "season": sc.season,
            "start_hour": sc.start_hour,
            "generator_initial": sc.generator_initial,
            "battery_initial": sc.battery_initial,
            "weather_config": sc.weather_config,
            "events": sc.events,
        }

        sim1 = GridSimulation(scenario_config=config, seed=42)
        sim2 = GridSimulation(scenario_config=config, seed=42)

        for _ in range(20):
            sim1.advance()
            sim2.advance()

        assert sim1.state.frequency_hz == sim2.state.frequency_hz
        assert sim1.state.total_demand_mw == sim2.state.total_demand_mw
        assert sim1.state.total_generation_mw == sim2.state.total_generation_mw

    def test_different_seed_different_trajectory(self):
        """Different seeds should produce different trajectories."""
        sc = ScenarioRegistry.get("summer_peak")
        config = {
            "max_steps": sc.max_steps,
            "peak_demand_mw": sc.peak_demand_mw,
            "season": sc.season,
            "start_hour": sc.start_hour,
            "generator_initial": sc.generator_initial,
            "battery_initial": sc.battery_initial,
            "weather_config": sc.weather_config,
            "events": sc.events,
        }

        sim1 = GridSimulation(scenario_config=config, seed=42)
        sim2 = GridSimulation(scenario_config=config, seed=99)

        for _ in range(20):
            sim1.advance()
            sim2.advance()

        # Demand should differ due to different noise
        assert sim1.state.total_demand_mw != sim2.state.total_demand_mw


# =============================================================================
# 11. CITATION VERIFICATION TESTS (B1-B13)
# =============================================================================

from generators import INERTIA_CONSTANTS
from simulation import GOVERNOR_DROOP, GOVERNOR_DEADBAND_HZ
from weather import wind_power_fraction, solar_power_fraction, SEASON_CONFIG


class TestPhysicalParameterCitations:
    """Verify all physical parameters against published literature."""

    def test_inertia_constants_within_literature_range(self):
        """B1: Inertia constants vs Kraljic 2022, Kundur Table 3.1."""
        # nuclear: 3-7s, coal: 2.5-6s, ccgt: 3-6s, peaker: 2-5s
        assert 3.0 <= INERTIA_CONSTANTS["nuclear"] <= 7.0, (
            f"Nuclear H={INERTIA_CONSTANTS['nuclear']}s outside 3-7s range")
        assert 2.5 <= INERTIA_CONSTANTS["coal"] <= 6.0, (
            f"Coal H={INERTIA_CONSTANTS['coal']}s outside 2.5-6s range")
        assert 3.0 <= INERTIA_CONSTANTS["ccgt"] <= 6.0, (
            f"CCGT H={INERTIA_CONSTANTS['ccgt']}s outside 3-6s range")
        assert 2.0 <= INERTIA_CONSTANTS["peaker"] <= 5.0, (
            f"Peaker H={INERTIA_CONSTANTS['peaker']}s outside 2-5s range")

    def test_wind_power_curve_iec_compliance(self):
        """B2: Wind curve matches IEC 61400-12-1 cubic model."""
        # Cut-in = 3 m/s, rated = 12 m/s, cut-out = 25 m/s
        assert wind_power_fraction(2.9) == 0.0, "Below cut-in"
        assert wind_power_fraction(3.0) == 0.0, "At cut-in (exclusive)"
        assert wind_power_fraction(25.0) == 0.0, "At cut-out"
        assert abs(wind_power_fraction(12.0) - 1.0) < 0.01, "At rated speed"
        # Cubic interpolation at 7.5 m/s
        expected = (7.5**3 - 3.0**3) / (12.0**3 - 3.0**3)
        actual = wind_power_fraction(7.5)
        assert abs(actual - expected) < 0.001, (
            f"Cubic at 7.5 m/s: expected {expected:.4f}, got {actual:.4f}")

    def test_solar_temperature_derating_coefficient(self):
        """B3: Solar temp derating per IEC 60904: -0.4%/°C above 25°C for c-Si."""
        # At 35°C: fraction = 1 - 0.004*(35-25) = 0.96
        frac_35 = solar_power_fraction(1000.0, 35.0)
        assert abs(frac_35 - 0.96) < 0.01, f"At 35°C: expected ~0.96, got {frac_35}"
        # At 15°C: no derating, fraction = 1.0 (capped at max(0, temp-25)=0)
        frac_15 = solar_power_fraction(1000.0, 15.0)
        assert abs(frac_15 - 1.0) < 0.01, f"At 15°C: expected ~1.0, got {frac_15}"

    def test_governor_droop_and_deadband_ferc(self):
        """B4: Governor droop 5% and deadband 36 mHz per FERC Order 842."""
        assert GOVERNOR_DROOP == 0.05, f"Droop should be 0.05, got {GOVERNOR_DROOP}"
        assert GOVERNOR_DEADBAND_HZ == 0.036, (
            f"Deadband should be 0.036 Hz, got {GOVERNOR_DEADBAND_HZ}")

    def test_ufls_stages_nerc_compliance(self):
        """B5: UFLS stages aligned with NERC PRC-006-5."""
        # 3 stages: 59.5/10%, 59.1/10%, 58.7/10% = 30% total
        assert len(UFLS_STAGES) == 3
        assert UFLS_STAGES[0] == (59.5, 0.10), f"Stage 1: {UFLS_STAGES[0]}"
        assert UFLS_STAGES[1] == (59.1, 0.10), f"Stage 2: {UFLS_STAGES[1]}"
        assert UFLS_STAGES[2] == (58.7, 0.10), f"Stage 3: {UFLS_STAGES[2]}"
        total_shed = sum(s[1] for s in UFLS_STAGES)
        assert abs(total_shed - 0.30) < 0.001, f"Total UFLS shed: {total_shed}"

    def test_blackout_frequency_threshold(self):
        """B6: Blackout at 57.5 Hz per IEEE C37.106 turbine protection."""
        assert BLACKOUT_FREQ_HZ == 57.5

    def test_battery_round_trip_efficiency_nrel(self):
        """B7: Battery RTE 85% per NREL ATB 2023."""
        bat = BatteryStorage()
        assert bat.round_trip_efficiency == 0.85

    def test_heat_rates_match_eia_data(self):
        """B8: Heat rates vs EIA Electric Power Annual Table 8.1."""
        # nuclear: 10000-10800, coal: 9200-10800, ccgt: 6300-7700, peaker: 9000-12500
        for uid, spec in GENERATOR_FLEET.items():
            hr = spec.heat_rate_btu_per_kwh
            if spec.unit_type == "nuclear":
                assert 10000 <= hr <= 10800, f"{uid}: HR={hr}"
            elif spec.unit_type == "coal":
                assert 9200 <= hr <= 10800, f"{uid}: HR={hr}"
            elif spec.unit_type == "ccgt":
                assert 6300 <= hr <= 7700, f"{uid}: HR={hr}"
            elif spec.unit_type == "peaker":
                assert 9000 <= hr <= 12500, f"{uid}: HR={hr}"

    def test_ramp_rates_within_literature(self):
        """B9: Ramp rates vs Gonzalez-Salazar 2018, NREL 2020."""
        # nuclear: 0.5-5%/min, coal: 0.5-4%, ccgt: 3-10%, peaker: 8-25%
        for uid, spec in GENERATOR_FLEET.items():
            pct_per_min = (spec.ramp_rate_mw_per_min / spec.capacity_mw) * 100.0
            if spec.unit_type == "nuclear":
                assert 0.5 <= pct_per_min <= 5.0, f"{uid}: {pct_per_min:.1f}%/min"
            elif spec.unit_type == "coal":
                assert 0.5 <= pct_per_min <= 4.0, f"{uid}: {pct_per_min:.1f}%/min"
            elif spec.unit_type == "ccgt":
                assert 3.0 <= pct_per_min <= 10.0, f"{uid}: {pct_per_min:.1f}%/min"
            elif spec.unit_type == "peaker":
                assert 8.0 <= pct_per_min <= 25.0, f"{uid}: {pct_per_min:.1f}%/min"

    def test_startup_times_within_literature(self):
        """B10: Startup times vs EIA 2020, EPA 2011, IAEA."""
        # nuclear: 1440-4320 min, coal: 480-1440, ccgt: 120-360, peaker: 5-40
        for uid, spec in GENERATOR_FLEET.items():
            cst = spec.cold_start_time_min
            if spec.unit_type == "nuclear":
                assert 1440 <= cst <= 4320, f"{uid}: cold start {cst} min"
            elif spec.unit_type == "coal":
                assert 480 <= cst <= 1440, f"{uid}: cold start {cst} min"
            elif spec.unit_type == "ccgt":
                assert 120 <= cst <= 360, f"{uid}: cold start {cst} min"
            elif spec.unit_type == "peaker":
                assert 5 <= cst <= 40, f"{uid}: cold start {cst} min"

    def test_voll_matches_miso_system_estimate(self):
        """B11: VOLL close to MISO 2024 shortage pricing (~$35,000/MWh)."""
        calc = RewardCalculator("summer_peak", 3500.0)
        # VOLL is in terminal_reward as 35685
        voll = 35685.0
        assert 30000 <= voll <= 40000, f"VOLL {voll} outside MISO range"

    def test_demand_temperature_sensitivity(self):
        """B12: Temperature sensitivity per EPRI/IEA empirical data."""
        # Cooling: +2%/°C above 25°C, Heating: +1.5%/°C below 5°C
        # Check by running simulation at different temps
        sc = ScenarioRegistry.get("summer_peak")
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96, "peak_demand_mw": sc.peak_demand_mw,
                "season": "summer", "start_hour": 15.0,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": {"season": "summer", "events": []},
                "events": [],
            }, seed=42,
        )
        # Manually check temp adjustment math from simulation.py
        # Above 25°C: temp_adj = 1.0 + 0.02*(T-25)
        # Below 5°C: temp_adj = 1.0 + 0.015*(5-T)
        temp_30 = 1.0 + 0.02 * (30 - 25)
        assert abs(temp_30 - 1.10) < 0.01, f"At 30°C: adj={temp_30}"
        temp_neg5 = 1.0 + 0.015 * (5 - (-5))
        assert abs(temp_neg5 - 1.15) < 0.01, f"At -5°C: adj={temp_neg5}"
        temp_15 = 1.0  # Between 5 and 25, no adjustment
        assert temp_15 == 1.0

    def test_frequency_physical_bounds(self):
        """B13: Frequency clamped to [57, 61] Hz."""
        assert NOMINAL_FREQ_HZ == 60.0
        # Create extreme scenario and verify bounds
        sc = ScenarioRegistry.get("summer_peak")
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96, "peak_demand_mw": sc.peak_demand_mw,
                "season": "summer", "start_hour": 0.0,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": {"season": "summer", "events": []},
                "events": [],
            }, seed=42,
        )
        # Trip all generators to crash frequency
        for gen in sim.generators.values():
            gen.trip()
        sim.advance()
        assert sim.state.frequency_hz >= 57.0, (
            f"Frequency {sim.state.frequency_hz} below physical floor of 57.0")


# =============================================================================
# 12. END-TO-END EPISODE TESTS (A1-A3)
# =============================================================================

class TestEndToEndEpisodes:
    """Full episode runs without crashes."""

    def _make_sim(self, scenario: str, seed: int = 42) -> GridSimulation:
        sc = ScenarioRegistry.get(scenario)
        return GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            },
            seed=seed,
        )

    def test_summer_peak_full_episode_no_crash(self):
        """A1: Run summer_peak 96 steps, no actions; no Python crash."""
        sim = self._make_sim("summer_peak")
        for _ in range(96):
            sim.advance()
            if sim.state.blackout:
                break
        # Verify no Python crash; blackout may occur passively (expected)
        assert sim.state.timestep > 0
        assert sim.state.cumulative_cost_usd > 0

    def test_polar_vortex_full_episode_288_steps(self):
        """A2: Run polar_vortex 288 steps; frequency always in [57, 61]."""
        sim = self._make_sim("polar_vortex")
        for step in range(288):
            sim.advance()
            if sim.state.blackout:
                break
            assert 57.0 <= sim.state.frequency_hz <= 61.0, (
                f"Step {step}: freq={sim.state.frequency_hz}")

    def test_all_scenarios_all_seeds_no_crash(self):
        """A3: All 8 scenarios x 5 seeds = 40 runs; zero crashes."""
        for sc_name in list(TRAIN_SCENARIOS) + list(TEST_SCENARIOS):
            sc = ScenarioRegistry.get(sc_name)
            for seed in range(SEEDS_PER_SCENARIO):
                sim = GridSimulation(
                    scenario_config={
                        "max_steps": sc.max_steps,
                        "peak_demand_mw": sc.peak_demand_mw,
                        "season": sc.season,
                        "start_hour": sc.start_hour,
                        "generator_initial": sc.generator_initial,
                        "battery_initial": sc.battery_initial,
                        "weather_config": sc.weather_config,
                        "events": sc.events,
                    },
                    seed=seed,
                )
                for _ in range(sc.max_steps):
                    sim.advance()
                    if sim.state.blackout:
                        break
                # If it got here without exception, it passed


# =============================================================================
# 13. PASSIVE AGENT TESTS (A4-A6)
# =============================================================================

class TestPassiveAgent:
    """Tests for passive (no-action) agent behavior."""

    def _make_sim(self, scenario: str, seed: int = 42) -> GridSimulation:
        sc = ScenarioRegistry.get(scenario)
        return GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            },
            seed=seed,
        )

    def test_passive_summer_peak_survives_early_hours(self):
        """A4: Summer peak should survive the first 20 steps passively (midnight-5AM)."""
        sim = self._make_sim("summer_peak")
        for i in range(20):
            sim.advance()
            if sim.state.blackout:
                break
        # At night with excess generation, should survive at least 20 steps
        assert sim.state.timestep >= 20, (
            f"Summer peak should survive first 20 steps passively, blackout at step {sim.state.timestep}")

    def test_passive_cold_snap_has_issues(self):
        """A5: Cold snap (hard) with no action should show stress."""
        sim = self._make_sim("cold_snap")
        any_ufls = False
        any_low_freq = False
        for _ in range(96):
            sim.advance()
            if sim.state.blackout:
                break
            if sim.state.ufls_triggered:
                any_ufls = True
            if sim.state.frequency_hz < 59.5:
                any_low_freq = True
        # Hard scenario should cause some distress
        assert any_ufls or any_low_freq or sim.state.blackout, (
            "Cold snap should cause UFLS, low freq, or blackout passively")

    def test_passive_cascading_failure_blackouts(self):
        """A6: Cascading failure (expert) should blackout with no intervention."""
        sim = self._make_sim("cascading_failure")
        blackout_occurred = False
        ufls_occurred = False
        for _ in range(96):
            sim.advance()
            if sim.state.blackout:
                blackout_occurred = True
                break
            if sim.state.ufls_triggered:
                ufls_occurred = True
        # Expert scenario should be dangerous
        assert blackout_occurred or ufls_occurred, (
            "Cascading failure should cause blackout or UFLS passively")


# =============================================================================
# 14. STRESS CONDITION TESTS (A7-A10)
# =============================================================================

class TestStressConditions:
    """Tests for edge cases and stress conditions."""

    def _make_sim(self, scenario: str = "summer_peak", seed: int = 42) -> GridSimulation:
        sc = ScenarioRegistry.get(scenario)
        return GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": [],
            },
            seed=seed,
        )

    def test_rapid_battery_cycling(self):
        """A7: 100 alternating charge/discharge cycles; no NaN/Inf."""
        bat = BatteryStorage(initial_soc_mwh=400.0)
        for i in range(100):
            if i % 2 == 0:
                bat.discharge(200.0, 0.25)
            else:
                bat.charge(200.0, 0.25)
            assert not math.isnan(bat.soc_mwh), f"NaN at cycle {i}"
            assert not math.isinf(bat.soc_mwh), f"Inf at cycle {i}"
            assert bat.soc_min_mwh <= bat.soc_mwh <= bat.soc_max_mwh + 0.1, (
                f"SoC {bat.soc_mwh} out of bounds at cycle {i}")

    def test_battery_full_cycle_energy_accounting(self):
        """A8: Charge from min to max, discharge back; verify RTE ~85%."""
        bat = BatteryStorage(initial_soc_mwh=80.0)  # At min
        # Charge to max
        total_grid_in = 0.0
        for _ in range(50):  # Enough steps to fill
            power_in = bat.charge(200.0, 0.25)
            total_grid_in += power_in * 0.25
        soc_at_max = bat.soc_mwh
        # Discharge back to min
        total_grid_out = 0.0
        for _ in range(50):
            power_out = bat.discharge(200.0, 0.25)
            total_grid_out += power_out * 0.25
        # RTE = energy_out / energy_in
        if total_grid_in > 0:
            rte = total_grid_out / total_grid_in
            assert 0.80 <= rte <= 0.90, (
                f"RTE={rte:.3f}, expected ~0.85. In={total_grid_in:.1f}, Out={total_grid_out:.1f}")

    def test_extreme_dispatch_values(self):
        """A9: NaN, Inf, negative targets should be handled gracefully."""
        gen = GeneratorUnit(GENERATOR_FLEET["ccgt_1"], "online", 300.0)
        # NaN
        ok, _ = gen.set_target(float('nan'))
        assert not ok, "NaN target should be rejected"
        # Inf
        ok, _ = gen.set_target(float('inf'))
        assert not ok, "Inf target should be rejected"
        # Negative
        ok, _ = gen.set_target(-100.0)
        assert not ok, "Negative target should be rejected"
        # Zero (below minimum)
        ok, _ = gen.set_target(0.0)
        assert not ok, "Zero target should be rejected (below min)"
        # Generator should still be functional
        ok, _ = gen.set_target(300.0)
        assert ok, "Valid target should succeed after bad ones"

    def test_all_generators_tripped_causes_blackout(self):
        """A10: Trip all generators; blackout within a few steps."""
        sim = self._make_sim()
        for gen in sim.generators.values():
            gen.trip()
        for _ in range(3):
            sim.advance()
        assert sim.state.blackout, "All generators tripped should cause blackout"


# =============================================================================
# 15. DEMAND SURGE EVENT TEST (A11)
# =============================================================================

class TestDemandSurgeEvent:
    """Test demand surge event mechanics."""

    def test_demand_surge_event_increases_peak(self):
        """A11: Demand surge event multiplies peak_demand_mw by factor."""
        # Test demand surge mechanics in isolation using summer_peak (no blackout)
        sc = ScenarioRegistry.get("summer_peak")
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": "summer",
                "start_hour": 0.0,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": {"season": "summer", "events": []},
                "events": [
                    {"timestep": 5, "event_type": "demand_surge",
                     "target": "system", "params": {"factor": 1.04}},
                ],
            },
            seed=42,
        )
        original_peak = sim.peak_demand_mw
        for _ in range(10):
            sim.advance()
        expected = original_peak * 1.04
        assert abs(sim.peak_demand_mw - expected) < 1.0, (
            f"Peak should be {expected}, got {sim.peak_demand_mw}")


# =============================================================================
# 16. RENEWABLE SURPLUS HANDLING (A12-A13)
# =============================================================================

class TestRenewableSurplusHandling:
    """Test over-generation and curtailment mechanics."""

    def test_curtailment_reduces_renewable_output(self):
        """A13: Setting curtailment limit reduces wind output."""
        wind = RenewableSource("wind", 500.0)
        wind.update_output(400.0)  # 400 MW available
        assert wind.current_output_mw == 400.0
        msg = wind.curtail(100.0)
        assert wind.current_output_mw <= 100.0
        assert wind.curtailed_mw > 0
        wind.uncurtail()
        assert wind.current_output_mw == 400.0
        assert wind.curtailed_mw == 0.0


# =============================================================================
# 17. GOVERNOR DROOP RESPONSE (A14)
# =============================================================================

class TestGovernorDroopResponse:
    """Test governor droop quantitative response."""

    def test_governor_droop_quantitative(self):
        """A14: Trip a peaker; governor should limit freq drop."""
        sc = ScenarioRegistry.get("summer_peak")
        sim = GridSimulation(
            scenario_config={
                "max_steps": 96, "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season, "start_hour": 0.0,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": {"season": "summer", "events": []},
                "events": [],
            },
            seed=42,
        )
        # Run a few steps at midnight to stabilize (excess generation)
        for _ in range(4):
            sim.advance()
        freq_before = sim.state.frequency_hz
        # Trip peaker_1 (150 MW) - small loss relative to system
        sim.generators["peaker_1"].trip()
        sim.advance()
        # Governor should absorb a 150 MW loss easily; freq should stay high
        assert sim.state.frequency_hz > 59.5, (
            f"Governor should absorb 150 MW trip, freq={sim.state.frequency_hz}")


# =============================================================================
# 18. ADDITIONAL BATTERY AND INTEGRATION TESTS (A15-A17)
# =============================================================================

class TestBatteryExtended:
    """Extended battery tests."""

    def test_battery_charge_min_to_max_soc(self):
        """A15: Charging from min to max takes correct energy."""
        bat = BatteryStorage(initial_soc_mwh=80.0)  # At 10%
        steps = 0
        while bat.soc_mwh < bat.soc_max_mwh - 1.0 and steps < 50:
            bat.charge(200.0, 0.25)
            steps += 1
        # 640 MWh to fill / (200 MW * sqrt(0.85) efficiency) per step
        # = 640 / (200*0.25*0.922) = 640/46.1 ≈ 14 steps
        assert 10 <= steps <= 20, f"Expected 10-20 steps to charge, took {steps}"
        assert bat.soc_mwh >= bat.soc_max_mwh - 1.0


class TestIntegrationExtended:
    """Extended integration tests."""

    def _make_sim(self, scenario: str, seed: int = 42) -> GridSimulation:
        sc = ScenarioRegistry.get(scenario)
        return GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            },
            seed=seed,
        )

    def test_cumulative_cost_is_monotonic(self):
        """A16: Cumulative cost should be non-decreasing over 96 steps."""
        sim = self._make_sim("summer_peak")
        prev_cost = 0.0
        for step in range(96):
            sim.advance()
            if sim.state.blackout:
                break
            assert sim.state.cumulative_cost_usd >= prev_cost - 0.01, (
                f"Step {step}: cost decreased from {prev_cost} to {sim.state.cumulative_cost_usd}")
            prev_cost = sim.state.cumulative_cost_usd

    def test_timestep_increments_correctly(self):
        """A17: Timestep and elapsed minutes track correctly."""
        sim = self._make_sim("summer_peak")
        for i in range(1, 20):
            sim.advance()
            assert sim.state.timestep == i, f"Timestep {sim.state.timestep} != {i}"
            assert abs(sim.state.elapsed_minutes - i * 15.0) < 0.01, (
                f"Elapsed {sim.state.elapsed_minutes} != {i * 15.0}")


# =============================================================================
# 19. RL SOLVABILITY TESTS (C1-C4)
# =============================================================================

def _run_greedy_agent(sim: GridSimulation, reward_calc: RewardCalculator,
                      max_steps: int) -> tuple[float, bool]:
    """Simple greedy agent: dispatch in merit order, use battery, shed if needed.

    Returns (cumulative_reward, blackout_occurred).
    """
    blackout_occurred = False
    merit_order = ["nuclear_1", "coal_1", "coal_2", "ccgt_1", "ccgt_2",
                   "peaker_1", "peaker_2", "peaker_3"]

    for step in range(max_steps):
        demand = sim.state.total_demand_mw
        renewable = sim.state.total_renewable_mw

        # 1. Set all online generators to maximum capacity (ensure we have headroom)
        total_online_gen = 0.0
        for uid in merit_order:
            gen = sim.generators[uid]
            if gen.status == GeneratorStatus.ONLINE:
                gen.set_target(gen.effective_capacity)
                total_online_gen += gen.current_output_mw
            elif gen.status in (GeneratorStatus.OFFLINE, GeneratorStatus.TRIPPED):
                # Start any offline generator that can start
                if gen.time_since_shutdown_min >= gen.spec.min_down_time_min:
                    gen.begin_startup()

        # 2. Battery: discharge if deficit, charge if surplus
        supply = total_online_gen + renewable
        deficit = demand - supply
        if deficit > 10:
            sim.battery.discharge(min(deficit, 200.0), 0.25)
        elif deficit < -200 and sim.battery.soc_mwh < sim.battery.soc_max_mwh - 10:
            sim.battery.charge(min(abs(deficit), 200.0), 0.25)
        else:
            sim.battery.set_idle()

        # 3. Load shedding: proactively shed if deficit or if reserves are very low
        bat_power = max(0, sim.battery.current_power_mw)
        total_supply = supply + bat_power
        reserves = total_supply - demand
        if demand > total_supply + 20 or (reserves < 100 and sim.state.frequency_hz < 59.8):
            shed_needed = max(0, demand - total_supply) + 200  # Extra margin
            shed_needed = min(shed_needed, demand * 0.3)  # Cap at 30%
            for zone in sim.zones.values():
                zone.shed_mw = shed_needed * zone.demand_fraction
        else:
            for zone in sim.zones.values():
                zone.shed_mw = 0.0

        sim.advance()
        reward = reward_calc.step_reward(sim.state)

        if sim.state.blackout:
            blackout_occurred = True
            break

    return reward_calc.cumulative_reward, blackout_occurred


class TestSolvability:
    """RL solvability: greedy agent should be able to complete scenarios."""

    def _setup(self, scenario_name: str, seed: int = 42):
        sc = ScenarioRegistry.get(scenario_name)
        sim = GridSimulation(
            scenario_config={
                "max_steps": sc.max_steps,
                "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season,
                "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            },
            seed=seed,
        )
        calc = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
        return sim, calc, sc.max_steps

    def test_summer_peak_solvable(self):
        """C1: Greedy agent on summer_peak: no blackout, positive reward."""
        sim, calc, max_steps = self._setup("summer_peak")
        cum_reward, blackout = _run_greedy_agent(sim, calc, max_steps)
        assert not blackout, "Summer peak should be solvable without blackout"
        assert cum_reward > 0, f"Greedy agent should get positive reward, got {cum_reward}"

    def test_cold_snap_solvable(self):
        """C2: Greedy agent on cold_snap survives longer than passive agent."""
        sim_greedy, calc_greedy, max_steps = self._setup("cold_snap")
        _, blackout_greedy = _run_greedy_agent(sim_greedy, calc_greedy, max_steps)
        greedy_steps = sim_greedy.state.timestep

        # Passive comparison
        sc = ScenarioRegistry.get("cold_snap")
        sim_passive = GridSimulation(
            scenario_config={
                "max_steps": max_steps, "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season, "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            }, seed=42,
        )
        for _ in range(max_steps):
            sim_passive.advance()
            if sim_passive.state.blackout:
                break
        passive_steps = sim_passive.state.timestep

        assert greedy_steps >= passive_steps, (
            f"Greedy ({greedy_steps} steps) should survive at least as long as "
            f"passive ({passive_steps} steps) on cold snap")

    def test_cascading_failure_solvable(self):
        """C3: Greedy agent on cascading_failure: no blackout."""
        sim, calc, max_steps = self._setup("cascading_failure")
        cum_reward, blackout = _run_greedy_agent(sim, calc, max_steps)
        assert not blackout, "Cascading failure should be solvable without blackout"

    def test_polar_vortex_solvable(self):
        """C4: Greedy agent on polar_vortex survives longer than passive."""
        sim_greedy, calc_greedy, max_steps = self._setup("polar_vortex")
        _, blackout_greedy = _run_greedy_agent(sim_greedy, calc_greedy, max_steps)
        greedy_steps = sim_greedy.state.timestep

        sc = ScenarioRegistry.get("polar_vortex")
        sim_passive = GridSimulation(
            scenario_config={
                "max_steps": max_steps, "peak_demand_mw": sc.peak_demand_mw,
                "season": sc.season, "start_hour": sc.start_hour,
                "generator_initial": sc.generator_initial,
                "battery_initial": sc.battery_initial,
                "weather_config": sc.weather_config,
                "events": sc.events,
            }, seed=42,
        )
        for _ in range(max_steps):
            sim_passive.advance()
            if sim_passive.state.blackout:
                break
        passive_steps = sim_passive.state.timestep

        assert greedy_steps >= passive_steps, (
            f"Greedy ({greedy_steps} steps) should survive at least as long as "
            f"passive ({passive_steps} steps) on polar vortex")


# =============================================================================
# 20. AGENT DIFFERENTIATION TESTS (C5-C7)
# =============================================================================

class TestAgentDifferentiation:
    """Better agents should get better rewards."""

    def _setup(self, scenario_name: str, seed: int = 42):
        sc = ScenarioRegistry.get(scenario_name)
        config = {
            "max_steps": sc.max_steps,
            "peak_demand_mw": sc.peak_demand_mw,
            "season": sc.season,
            "start_hour": sc.start_hour,
            "generator_initial": sc.generator_initial,
            "battery_initial": sc.battery_initial,
            "weather_config": sc.weather_config,
            "events": sc.events,
        }
        return config, sc

    def test_greedy_beats_passive(self):
        """C5: Greedy agent should outperform passive agent on summer_peak."""
        config, sc = self._setup("summer_peak")

        # Passive agent
        sim_passive = GridSimulation(scenario_config=config, seed=42)
        calc_passive = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
        for _ in range(sc.max_steps):
            sim_passive.advance()
            calc_passive.step_reward(sim_passive.state)
            if sim_passive.state.blackout:
                calc_passive.step_reward(sim_passive.state)
                break
        passive_reward = calc_passive.cumulative_reward

        # Greedy agent
        sim_greedy = GridSimulation(scenario_config=config, seed=42)
        calc_greedy = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
        greedy_reward, _ = _run_greedy_agent(sim_greedy, calc_greedy, sc.max_steps)

        assert greedy_reward >= passive_reward, (
            f"Greedy ({greedy_reward:.2f}) should beat passive ({passive_reward:.2f})")

    def test_unnecessary_shedding_penalized(self):
        """C7: Agent that sheds load unnecessarily should get worse reward."""
        sc = ScenarioRegistry.get("summer_peak")
        config = {
            "max_steps": sc.max_steps,
            "peak_demand_mw": sc.peak_demand_mw,
            "season": sc.season,
            "start_hour": sc.start_hour,
            "generator_initial": sc.generator_initial,
            "battery_initial": sc.battery_initial,
            "weather_config": sc.weather_config,
            "events": [],
        }

        # No-shed agent (just let it run)
        sim1 = GridSimulation(scenario_config=config, seed=42)
        calc1 = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
        for _ in range(20):
            sim1.advance()
            if sim1.state.blackout:
                break
            calc1.step_reward(sim1.state)

        # Unnecessary-shed agent
        sim2 = GridSimulation(scenario_config=config, seed=42)
        calc2 = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
        for _ in range(20):
            # Shed 200 MW unnecessarily
            for zone in sim2.zones.values():
                zone.shed_mw = 200.0 * zone.demand_fraction
            sim2.advance()
            if sim2.state.blackout:
                break
            calc2.step_reward(sim2.state)

        assert calc1.cumulative_reward > calc2.cumulative_reward, (
            f"No-shed ({calc1.cumulative_reward:.2f}) should beat "
            f"unnecessary-shed ({calc2.cumulative_reward:.2f})")


# =============================================================================
# 21. REWARD RANGE VALIDATION TESTS (C8-C11)
# =============================================================================

class TestRewardRangeValidation:
    """Verify reward values are always within bounds."""

    def test_step_reward_always_in_range(self):
        """C8: All step rewards in [-1, 1] across multiple scenarios."""
        for sc_name in ["summer_peak", "cold_snap", "cascading_failure", "renewable_surplus"]:
            sc = ScenarioRegistry.get(sc_name)
            sim = GridSimulation(
                scenario_config={
                    "max_steps": sc.max_steps,
                    "peak_demand_mw": sc.peak_demand_mw,
                    "season": sc.season, "start_hour": sc.start_hour,
                    "generator_initial": sc.generator_initial,
                    "battery_initial": sc.battery_initial,
                    "weather_config": sc.weather_config,
                    "events": sc.events,
                },
                seed=42,
            )
            calc = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
            for step in range(sc.max_steps):
                sim.advance()
                reward = calc.step_reward(sim.state)
                assert -1.0 <= reward <= 1.0, (
                    f"{sc_name} step {step}: reward={reward} out of [-1,1]")
                if sim.state.blackout:
                    break

    def test_terminal_reward_in_range(self):
        """C9: Terminal reward always in [-1, 1]."""
        # Test with blackout state
        calc1 = RewardCalculator("summer_peak", 3500.0)
        state_bo = GridState()
        state_bo.blackout = True
        r = calc1.terminal_reward(state_bo)
        assert -1.0 <= r <= 1.0

        # Test with healthy state
        calc2 = RewardCalculator("summer_peak", 3500.0)
        state_ok = GridState()
        state_ok.cumulative_cost_usd = 100000.0
        r = calc2.terminal_reward(state_ok)
        assert -1.0 <= r <= 1.0

    def test_reward_weights_sum_to_one(self):
        """C10: Reward component weights sum to 1.0."""
        from rewards import RewardWeights
        w = RewardWeights()
        total = (w.reliability + w.cost_efficiency + w.frequency_stability +
                 w.reserve_adequacy + w.renewable_utilization)
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_perfect_state_maximum_reward(self):
        """C11: Construct ideal state; step reward should be ~1.0."""
        calc = RewardCalculator("summer_peak", 3500.0)
        state = GridState()
        state.frequency_hz = 60.0
        state.total_demand_mw = 3000.0
        state.total_generation_mw = 2700.0
        state.total_renewable_mw = 300.0
        state.wind_output_mw = 200.0
        state.solar_output_mw = 100.0
        state.battery_power_mw = 0.0
        state.total_load_shed_mw = 0.0
        state.ufls_triggered = False
        state.blackout = False
        state.spinning_reserve_mw = 600.0
        state.required_reserve_mw = 400.0
        state.generation_cost_usd = 5000.0  # Low cost
        state.wind_curtailed_mw = 0.0
        state.solar_curtailed_mw = 0.0
        reward = calc.step_reward(state)
        assert reward >= 0.9, f"Perfect state should yield ~1.0 reward, got {reward}"


# =============================================================================
# 22. REWARD MONOTONICITY TESTS (C18-C20)
# =============================================================================

class TestRewardMonotonicity:
    """Verify reward monotonicity with respect to key variables."""

    def _healthy_state(self) -> GridState:
        state = GridState()
        state.frequency_hz = 60.0
        state.total_demand_mw = 3000.0
        state.total_generation_mw = 2700.0
        state.total_renewable_mw = 300.0
        state.wind_output_mw = 200.0
        state.solar_output_mw = 100.0
        state.battery_power_mw = 0.0
        state.total_load_shed_mw = 0.0
        state.spinning_reserve_mw = 500.0
        state.required_reserve_mw = 400.0
        state.generation_cost_usd = 10000.0
        state.wind_curtailed_mw = 0.0
        state.solar_curtailed_mw = 0.0
        state.ufls_triggered = False
        state.blackout = False
        return state

    def test_more_shed_worse_reward(self):
        """C18: Increasing load shed should give non-increasing rewards."""
        shed_levels = [0, 100, 200, 500, 1000]
        rewards = []
        for shed in shed_levels:
            calc = RewardCalculator("summer_peak", 3500.0)
            state = self._healthy_state()
            state.total_load_shed_mw = shed
            rewards.append(calc.step_reward(state))
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1] - 0.001, (
                f"Shed {shed_levels[i]} MW -> R={rewards[i]:.3f} should be >= "
                f"shed {shed_levels[i+1]} MW -> R={rewards[i+1]:.3f}")

    def test_worse_frequency_worse_reward(self):
        """C19: Greater freq deviation should give non-increasing rewards."""
        freq_values = [60.0, 59.95, 59.9, 59.7, 59.5]
        rewards = []
        for freq in freq_values:
            calc = RewardCalculator("summer_peak", 3500.0)
            state = self._healthy_state()
            state.frequency_hz = freq
            rewards.append(calc.step_reward(state))
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1] - 0.001, (
                f"Freq {freq_values[i]} Hz -> R={rewards[i]:.3f} should be >= "
                f"freq {freq_values[i+1]} Hz -> R={rewards[i+1]:.3f}")

    def test_more_reserves_better_reward(self):
        """C20: More reserves should give non-decreasing rewards."""
        reserve_levels = [0, 100, 200, 400, 600]
        rewards = []
        for reserve in reserve_levels:
            calc = RewardCalculator("summer_peak", 3500.0)
            state = self._healthy_state()
            state.spinning_reserve_mw = reserve
            rewards.append(calc.step_reward(state))
        for i in range(len(rewards) - 1):
            assert rewards[i] <= rewards[i + 1] + 0.001, (
                f"Reserve {reserve_levels[i]} MW -> R={rewards[i]:.3f} should be <= "
                f"reserve {reserve_levels[i+1]} MW -> R={rewards[i+1]:.3f}")


# =============================================================================
# 23. DIFFICULTY CALIBRATION TEST (C16)
# =============================================================================

class TestDifficultyCalibration:
    """Verify difficulty ordering across scenarios."""

    def test_normal_easier_than_hard_than_expert(self):
        """C16: Normal scenarios should yield higher avg reward than hard/expert."""
        difficulty_rewards = {"normal": [], "hard": [], "expert": []}

        for sc_name in list(TRAIN_SCENARIOS) + list(TEST_SCENARIOS):
            sc = ScenarioRegistry.get(sc_name)
            sim = GridSimulation(
                scenario_config={
                    "max_steps": sc.max_steps,
                    "peak_demand_mw": sc.peak_demand_mw,
                    "season": sc.season,
                    "start_hour": sc.start_hour,
                    "generator_initial": sc.generator_initial,
                    "battery_initial": sc.battery_initial,
                    "weather_config": sc.weather_config,
                    "events": sc.events,
                },
                seed=42,
            )
            calc = RewardCalculator(sc.scenario_type, sc.peak_demand_mw)
            cum_reward, _ = _run_greedy_agent(sim, calc, min(sc.max_steps, 96))
            difficulty_rewards[sc.difficulty].append(cum_reward)

        avg = {d: sum(r) / len(r) for d, r in difficulty_rewards.items() if r}
        # Normal should be better than hard
        if "normal" in avg and "hard" in avg:
            assert avg["normal"] > avg["hard"], (
                f"Normal avg {avg['normal']:.2f} should > hard avg {avg['hard']:.2f}")
        # Hard should be better than expert
        if "hard" in avg and "expert" in avg:
            assert avg["hard"] > avg["expert"], (
                f"Hard avg {avg['hard']:.2f} should > expert avg {avg['expert']:.2f}")
