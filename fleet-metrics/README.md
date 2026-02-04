# Fleet Performance & Behavior Metrics Analysis (Simulation-Based)

## Goal
Build a Zoox-style evaluation pipeline that:
- analyzes driving time-series logs,
- detects behavior & safety events (lane changes, pullovers, harsh braking, near-collisions),
- computes fleet KPIs (miles-per-disengagement, events per 100 miles),
- clusters driving behavior patterns,
- stores results in SQLite for SQL analytics.

## Why this matches Zoox
Zoox Data Science teams (Autonomy V&V / Core DS) measure autonomy performance using real-world or simulation logs:
- fleet metrics & regressions
- behavior clustering (lane changes / pullovers)
- safety-critical events
- SQL + dashboards for monitoring

This project demonstrates the same workflow end-to-end.

## What the dataset is
This project generates a synthetic “simulation” dataset:
- multiple runs (scenarios)
- multiple vehicles per run
- per-timestep trajectories (x,y, lane, speed, acceleration)
- lead vehicle relationships to estimate TTC (time-to-collision)

You can later swap in a real public dataset (e.g., NGSIM) without changing the analysis steps.

## Key Metrics
- Miles per disengagement (MPD)
- Lane changes / 100 miles
- Harsh brakes / 100 miles
- Near collisions / 100 miles (TTC <= threshold)
- Pullovers / 100 miles

## Outputs
- `outputs/fleet.db` SQLite database with:
  - `per_step_logs` (time-series)
  - `vehicle_metrics` (aggregated per vehicle)
- `outputs/figures/*.png` plots for distributions and cluster summaries

## Run
pip install -r requirements.txt
python fleet_simulation_analysis.py

## Example SQL questions you can answer
- Which vehicles have the most near-collision events?
- How do event rates vary per run (scenario)?
- Which scenarios regress in MPD compared to baseline?
