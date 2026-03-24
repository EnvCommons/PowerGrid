"""Reward calculation for the power grid environment.

Dense, multi-component reward calculated each timestep:
- Reliability (40%): Penalty for unserved energy
- Cost efficiency (25%): Lower generation cost is better
- Frequency stability (15%): Penalty for deviation from 60 Hz
- Reserve adequacy (10%): Penalty if reserves below requirement
- Renewable utilization (10%): Bonus for using renewables efficiently
"""

from __future__ import annotations

from dataclasses import dataclass

from simulation import GridState, NOMINAL_FREQ_HZ, GOVERNOR_DEADBAND_HZ


# =============================================================================
# Reward weights
# =============================================================================

@dataclass(frozen=True)
class RewardWeights:
    reliability: float = 0.40
    cost_efficiency: float = 0.25
    frequency_stability: float = 0.15
    reserve_adequacy: float = 0.10
    renewable_utilization: float = 0.10


# =============================================================================
# Reward Calculator
# =============================================================================

class RewardCalculator:
    """Calculates dense per-step and terminal rewards."""

    def __init__(self, scenario_type: str, peak_demand_mw: float) -> None:
        self.weights = RewardWeights()
        self.scenario_type = scenario_type
        self.peak_demand_mw = peak_demand_mw
        self.baseline_cost_per_mwh = 40.0  # Typical average for comparison
        self.cumulative_shed_mwh = 0.0
        self.consecutive_shed_steps = 0
        self.cumulative_reward = 0.0

    def step_reward(self, state: GridState, dt_hours: float = 0.25) -> float:
        """Calculate reward for one timestep."""
        reward = 0.0

        # --- 1. RELIABILITY (40%): Penalty for unserved energy ---
        if state.total_load_shed_mw > 0:
            shed_fraction = state.total_load_shed_mw / max(state.total_demand_mw, 1.0)
            reliability_score = max(0.0, 1.0 - shed_fraction * 10.0)
        else:
            reliability_score = 1.0
        reward += self.weights.reliability * reliability_score

        # --- 2. COST EFFICIENCY (25%): Lower cost is better ---
        total_energy = max(state.total_generation_mw * dt_hours, 1.0)
        actual_cost_per_mwh = state.generation_cost_usd / total_energy
        cost_ratio = actual_cost_per_mwh / max(self.baseline_cost_per_mwh, 1.0)
        cost_score = max(0.0, min(1.0, 2.0 - cost_ratio))
        reward += self.weights.cost_efficiency * cost_score

        # --- 3. FREQUENCY STABILITY (15%): Penalty for deviation ---
        freq_dev = abs(state.frequency_hz - NOMINAL_FREQ_HZ)
        if freq_dev <= GOVERNOR_DEADBAND_HZ:
            freq_score = 1.0
        elif freq_dev < 0.5:
            freq_score = max(0.0, 1.0 - (freq_dev - GOVERNOR_DEADBAND_HZ) / (0.5 - GOVERNOR_DEADBAND_HZ))
        else:
            freq_score = 0.0
        reward += self.weights.frequency_stability * freq_score

        # --- 4. RESERVE ADEQUACY (10%): Penalty if below requirement ---
        if state.required_reserve_mw > 0:
            reserve_ratio = state.spinning_reserve_mw / state.required_reserve_mw
            reserve_score = min(1.0, max(0.0, reserve_ratio))
        else:
            reserve_score = 1.0
        reward += self.weights.reserve_adequacy * reserve_score

        # --- 5. RENEWABLE UTILIZATION (10%): Bonus for using renewables ---
        total_available_re = (state.wind_output_mw + state.solar_output_mw +
                              state.wind_curtailed_mw + state.solar_curtailed_mw)
        if total_available_re > 0:
            re_used = (state.wind_output_mw + state.solar_output_mw) / total_available_re
            renewable_score = re_used
        else:
            renewable_score = 1.0
        reward += self.weights.renewable_utilization * renewable_score

        # === PENALTIES ===

        # Blackout: catastrophic
        if state.blackout:
            reward = -1.0
            self._track_shed(state, dt_hours)
            self.cumulative_reward += reward
            return reward

        # UFLS activation penalty
        if state.ufls_triggered:
            reward -= 0.2

        # Political penalty for sustained shedding (> 2 hours = 8 steps)
        if state.total_load_shed_mw > 0:
            self.consecutive_shed_steps += 1
            if self.consecutive_shed_steps > 8:
                reward -= 0.1
        else:
            self.consecutive_shed_steps = 0

        reward = max(-1.0, min(1.0, reward))

        # Track cumulative unserved energy
        self._track_shed(state, dt_hours)
        self.cumulative_reward += reward

        return reward

    def _track_shed(self, state: GridState, dt_hours: float) -> None:
        if state.total_load_shed_mw > 0:
            self.cumulative_shed_mwh += state.total_load_shed_mw * dt_hours

    def terminal_reward(self, state: GridState) -> float:
        """Calculate terminal reward at end of episode."""
        if state.blackout:
            return -1.0

        # Base: survived
        base = 0.5

        # Cost efficiency bonus: lower cumulative cost = better
        # Normalize by expected cost (peak_demand * 96 steps * 0.25h * $40/MWh)
        expected_cost = self.peak_demand_mw * 96 * 0.25 * self.baseline_cost_per_mwh
        if expected_cost > 0:
            cost_ratio = state.cumulative_cost_usd / expected_cost
            cost_bonus = max(0.0, min(0.3, 0.3 * (2.0 - cost_ratio)))
        else:
            cost_bonus = 0.15

        # Unserved energy penalty
        # VOLL = $35,685/MWh, normalize to [0, 0.5]
        voll = 35685.0
        unserved_cost = self.cumulative_shed_mwh * voll
        unserved_penalty = min(0.5, unserved_cost / 1e7)

        terminal = base + cost_bonus - unserved_penalty
        return max(-1.0, min(1.0, terminal))

    def is_terminal(self, state: GridState, step: int, max_steps: int) -> tuple[bool, str]:
        """Check if episode should end."""
        if state.blackout:
            return True, "blackout"
        if state.frequency_hz < 57.5:
            return True, "frequency_collapse"
        if step >= max_steps:
            return True, "completed"
        return False, ""

    def get_reward_breakdown(self, state: GridState, dt_hours: float = 0.25) -> dict:
        """Get detailed reward breakdown for metadata."""
        # Reliability
        if state.total_load_shed_mw > 0:
            shed_fraction = state.total_load_shed_mw / max(state.total_demand_mw, 1.0)
            reliability_score = max(0.0, 1.0 - shed_fraction * 10.0)
        else:
            reliability_score = 1.0

        # Cost
        total_energy = max(state.total_generation_mw * dt_hours, 1.0)
        actual_cost_per_mwh = state.generation_cost_usd / total_energy
        cost_ratio = actual_cost_per_mwh / max(self.baseline_cost_per_mwh, 1.0)
        cost_score = max(0.0, min(1.0, 2.0 - cost_ratio))

        # Frequency
        freq_dev = abs(state.frequency_hz - NOMINAL_FREQ_HZ)
        if freq_dev <= GOVERNOR_DEADBAND_HZ:
            freq_score = 1.0
        elif freq_dev < 0.5:
            freq_score = max(0.0, 1.0 - (freq_dev - GOVERNOR_DEADBAND_HZ) / (0.5 - GOVERNOR_DEADBAND_HZ))
        else:
            freq_score = 0.0

        # Reserve
        if state.required_reserve_mw > 0:
            reserve_score = min(1.0, max(0.0, state.spinning_reserve_mw / state.required_reserve_mw))
        else:
            reserve_score = 1.0

        # Renewable
        total_available_re = (state.wind_output_mw + state.solar_output_mw +
                              state.wind_curtailed_mw + state.solar_curtailed_mw)
        if total_available_re > 0:
            renewable_score = (state.wind_output_mw + state.solar_output_mw) / total_available_re
        else:
            renewable_score = 1.0

        return {
            "reliability": round(reliability_score, 3),
            "cost_efficiency": round(cost_score, 3),
            "frequency_stability": round(freq_score, 3),
            "reserve_adequacy": round(reserve_score, 3),
            "renewable_utilization": round(renewable_score, 3),
            "cumulative_shed_mwh": round(self.cumulative_shed_mwh, 1),
            "consecutive_shed_steps": self.consecutive_shed_steps,
        }
