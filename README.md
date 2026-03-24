# PowerGrid

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/PowerGrid)

## Description

PowerGrid is a power grid environment where agents dispatch generators, manage battery storage, handle renewable variability, and maintain grid frequency across crisis scenarios inspired by the 2021 Texas winter storm, the 2003 Northeast blackout, and the 2016 South Australia blackout.

Note: this is a synthetic environment which is majority AI-generated; we recommend testing thoroughly before integrating into an RL pipeline.

## Capabilities

- Economic dispatch optimization across 8 thermal generators with quadratic cost curves
- Frequency regulation via governor droop response and under-frequency load shedding
- Grid-scale battery storage management (200 MW / 800 MWh, 85% round-trip efficiency)
- Renewable integration (500 MW wind, 300 MW solar) with curtailment decisions
- Emergency load shedding and restoration across 3 transmission zones
- Transmission congestion management with N-1 contingency constraints
- Multi-day crisis management (up to 72 hours in polar vortex scenario)
- Dense, multi-component reward signal across 5 dimensions

## License

MIT

## Tasks

There are 4 training scenarios (5 seeds each = 20 training tasks):

- **summer_peak**: Normal hot summer day dispatch optimization. Evening ramp challenge as solar fades and AC load peaks.
- **wind_drought**: Wind drops from 80% to 5% capacity over 2 hours. Tests proactive thermal ramp-up and reserve management.
- **cold_snap**: Extreme cold (-20C), demand surges to 5,250 MW, gas supply curtailed, generator trips. Inspired by the February 2021 Texas winter storm.
- **line_outage**: Major transmission line trips followed by a generator trip (N-1-1 contingency). Tests transmission-aware redispatch.

And 4 test scenarios (5 seeds each = 20 test tasks):

- **cascading_failure**: Sequential line and generator trips leading to frequency instability. Inspired by the August 2003 Northeast blackout.
- **renewable_surplus**: Low demand weekend with excessive wind and solar. Tests minimum generation management and frequency stability with low inertia.
- **polar_vortex**: 72-hour multi-day extreme cold event with progressive generator deratings and trips. Tests long-horizon strategic planning.
- **price_spike_crisis**: Extreme heat wave drives demand beyond capacity. Political pressure limits acceptable load shedding duration.

Each 24-hour scenario has 96 timesteps (15 minutes each). The polar_vortex scenario has 288 timesteps (72 hours).

## Reward Structure

This is a dense, verifiable reward environment. Rewards are calculated per timestep as a weighted sum of five components:

- **Reliability** (40%): Penalty for unserved energy (load shedding)
- **Cost Efficiency** (25%): Lower generation cost relative to baseline
- **Frequency Stability** (15%): Penalty for frequency deviation from 60 Hz
- **Reserve Adequacy** (10%): Penalty if spinning reserves fall below NERC requirement
- **Renewable Utilization** (10%): Bonus for using available renewables without curtailment

Terminal reward of -1.0 for total blackout (frequency collapse below 57.5 Hz). We do not use LLM graders.

## Tools

Agents have 11 tools:

| Tool | Time Advance | Description |
|------|:---:|-------------|
| `observe_grid` | No | Read full grid state: frequency, demand, generation, reserves, weather, costs |
| `dispatch_generators` | Yes | Set MW output targets for one or more generators |
| `control_battery` | Yes | Charge, discharge, or idle the 200 MW battery |
| `manage_reserves` | Yes | Set spinning reserve target (advisory) |
| `shed_load` | Yes | Emergency load shedding by zone (last resort) |
| `restore_load` | Yes | Restore previously shed load |
| `start_generator` | Yes | Begin startup of an offline unit |
| `stop_generator` | Yes | Begin shutdown of an online unit |
| `curtail_renewable` | Yes | Limit wind or solar output |
| `advance_time` | Yes | Move to next 15-minute timestep |
| `submit_log` | No | Document reasoning (no simulation effect) |

## Time Horizon

Each scenario runs for 96 timesteps (24 hours) except for the polar_vortex scenario which runs for 288 timesteps (72 hours). Each timestep represents 15 minutes of simulated time.

## Other Environment Requirements

There are no further environment requirements; PowerGrid works out of the box with the OpenReward endpoint without any external secrets.

## Safety

Agents in PowerGrid are tasked with operating a power grid simulation where their decisions affect the reliability of electricity supply to ~2 million simulated customers. The environment does not present direct real-world safety risks as all interactions occur within a self-contained simulation. The environment teaches agents to balance economic efficiency against reliability, with heavy penalties for blackouts and load shedding, which aligns with responsible grid operation practices.

## Citations

```bibtex
@dataset{GRPowerGrid,
  author    = {General Reasoning Inc. Team},
  title     = {PowerGrid},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/PowerGrid}
}
```
