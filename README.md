# Fleet Performance & Behavior Metrics Analysis  
### Simulation-Based Evaluation of Autonomous Vehicle Systems

---

## Overview
Autonomous vehicle (AV) teams rely on large-scale data analysis to evaluate **safety, comfort, and performance** across their fleet. This project implements an **end-to-end, simulation-based autonomy evaluation pipeline** that analyzes time-series driving data, detects safety-critical and behavioral events, computes fleet-level metrics, and enables SQL-based analytics.

The project is designed to closely mirror how **Autonomy Software V&V** and **Core Data Science** teams evaluate autonomous driving systems using real-world or simulation logs.

---

## Why This Project Matters
Autonomous systems must be evaluated across thousands of scenarios before deployment. This requires:
- Reliable **performance metrics**
- Detection of **safety-critical events**
- Analysis of **driving behavior patterns**
- Scalable **data pipelines** and **metrics platforms**

This project demonstrates how data science enables **data-driven decision-making** in autonomy development.

---

## What This Project Does
This project builds a full evaluation workflow that:

1. Generates simulation-style fleet driving data  
2. Detects driving behavior and safety events  
3. Computes fleet-level autonomy KPIs  
4. Clusters driving behavior patterns  
5. Stores results in a SQL-accessible metrics database  

---

## Dataset
### Synthetic Simulation Fleet Logs
The dataset is **synthetically generated** to resemble simulation or logged fleet data:

Each timestep includes:
- Vehicle position (`x`, `y`)
- Lane ID
- Speed and acceleration
- Lead-vehicle relationship (for safety analysis)

The dataset supports:
- Multi-vehicle scenarios
- Multiple simulation runs
- Time-series analysis
- Safety and behavior detection

> Note: The same pipeline can be applied to real public datasets (e.g., NGSIM) without changing the analysis logic.

---

## Driving Events Detected
From raw time-series logs, the pipeline automatically detects:

- **Lane changes**
- **Pullovers** (vehicle stops on the shoulder)
- **Harsh braking** (comfort & safety proxy)
- **Near-collision events** using **Time-to-Collision (TTC)**
- **Disengagement events** (triggered by severe safety signals)

### Safety Logic
- **TTC (Time-to-Collision)** is computed using lead-vehicle distance and relative speed
- TTC below a threshold is flagged as a near-collision
- Extremely low TTC or aggressive braking triggers a disengagement event

---

## Fleet-Level Metrics (KPIs)
The following autonomy metrics are computed:

- **Total miles driven**
- **Miles per disengagement (MPD)**
- **Lane changes per 100 miles**
- **Harsh brakes per 100 miles**
- **Near-collisions per 100 miles**
- **Pullovers per 100 miles**

These metrics are commonly used to track **system performance, regressions, and safety trends**.

---

## Behavior Analysis & Clustering
Vehicles are clustered based on behavioral features such as:
- Lane change frequency
- Harsh braking rate
- Near-collision rate
- Pullovers
- Average speed

This enables:
- Identification of driving behavior archetypes
- Detection of anomalous or risky behaviors
- Debugging of system regressions

---

## Metrics Platform & SQL Analytics
All processed data is stored in a **SQLite database**, enabling fast analysis using SQL.

### Tables
- `per_step_logs` — raw time-series driving data
- `vehicle_metrics` — aggregated metrics per vehicle

### Example SQL Queries
```sql
-- Vehicles with the highest number of near-collision events
SELECT run_id, vehicle_id, near_collision_events
FROM vehicle_metrics
ORDER BY near_collision_events DESC
LIMIT 10;

-- Average vehicle speed per simulation run
SELECT run_id, AVG(avg_speed) AS avg_speed
FROM vehicle_metrics
GROUP BY run_id
ORDER BY avg_speed DESC;
