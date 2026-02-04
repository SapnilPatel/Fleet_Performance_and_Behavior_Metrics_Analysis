import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass
class SimConfig:
    seed: int = 7
    dt: float = 0.1                   # seconds
    sim_seconds: int = 600            # 10 min per run
    n_runs: int = 10                  # "scenarios"
    vehicles_per_run: int = 40
    lane_width_m: float = 3.7
    road_lanes: int = 4
    base_speed_mps: float = 22.0      # ~49 mph
    speed_noise: float = 2.5
    accel_noise: float = 0.7
    lane_change_prob_per_min: float = 1.2
    pullover_prob_per_min: float = 0.15
    harsh_brake_threshold: float = -3.0   # m/s^2
    ttc_threshold: float = 2.0           # seconds (near collision)
    disengagement_ttc_trigger: float = 1.2
    disengagement_harsh_brake_trigger: float = -4.5


# -----------------------------
# Synthetic "simulation" dataset
# -----------------------------
def generate_synthetic_fleet_logs(cfg: SimConfig) -> pd.DataFrame:
    """
    Generates time-series logs for multiple runs.
    Columns:
      run_id, t, vehicle_id, x, y, lane_id, v, a, lead_id
    """
    rng = np.random.default_rng(cfg.seed)
    rows = []

    steps = int(cfg.sim_seconds / cfg.dt)

    for run_id in range(cfg.n_runs):
        # initialize vehicles with random positions on x (meters), lanes, speeds
        vehicle_ids = np.arange(cfg.vehicles_per_run)
        lane_id = rng.integers(0, cfg.road_lanes, size=cfg.vehicles_per_run)
        y = lane_id * cfg.lane_width_m + cfg.lane_width_m / 2.0

        # spread vehicles along x with jitter
        x = np.sort(rng.uniform(0, 2000, size=cfg.vehicles_per_run))
        v = np.clip(rng.normal(cfg.base_speed_mps, cfg.speed_noise, size=cfg.vehicles_per_run), 5, 40)
        a = rng.normal(0, cfg.accel_noise, size=cfg.vehicles_per_run)

        # per-vehicle state for lane change / pullover
        target_lane = lane_id.copy()
        pullover_active = np.zeros(cfg.vehicles_per_run, dtype=bool)

        # lane-change probability per step
        lc_prob = (cfg.lane_change_prob_per_min / 60.0) * cfg.dt
        po_prob = (cfg.pullover_prob_per_min / 60.0) * cfg.dt

        for k in range(steps):
            t = k * cfg.dt

            # decide lane changes
            do_lc = rng.random(cfg.vehicles_per_run) < lc_prob
            for i in np.where(do_lc)[0]:
                if pullover_active[i]:
                    continue
                # attempt move -1 or +1 lane within bounds
                move = rng.choice([-1, 1])
                new_lane = int(np.clip(target_lane[i] + move, 0, cfg.road_lanes - 1))
                target_lane[i] = new_lane

            # decide pullovers: vehicle moves to rightmost lane and decelerates to stop
            do_po = (rng.random(cfg.vehicles_per_run) < po_prob) & (~pullover_active)
            for i in np.where(do_po)[0]:
                pullover_active[i] = True
                target_lane[i] = cfg.road_lanes - 1  # rightmost

            # smooth lane transitions
            desired_y = target_lane * cfg.lane_width_m + cfg.lane_width_m / 2.0
            y += (desired_y - y) * 0.08  # lateral smoothing

            # longitudinal dynamics
            # base accel + noise
            a = rng.normal(0, cfg.accel_noise, size=cfg.vehicles_per_run)

            # if pullover active, decelerate and stop
            a[pullover_active] += -1.0
            v = np.clip(v + a * cfg.dt, 0, 45)
            v[pullover_active] = np.clip(v[pullover_active], 0, 8)  # stop-ish

            # update x
            x = x + v * cfg.dt

            # assign lane id from y
            lane_id = np.clip((y // cfg.lane_width_m).astype(int), 0, cfg.road_lanes - 1)

            # find "lead vehicle" within same lane (simple)
            lead_id = np.full(cfg.vehicles_per_run, -1, dtype=int)
            for ln in range(cfg.road_lanes):
                idx = np.where(lane_id == ln)[0]
                if len(idx) == 0:
                    continue
                # sort by x
                order = idx[np.argsort(x[idx])]
                for j in range(len(order) - 1):
                    follower = order[j]
                    leader = order[j + 1]
                    lead_id[follower] = leader

            for i in range(cfg.vehicles_per_run):
                rows.append((run_id, t, int(vehicle_ids[i]), float(x[i]), float(y[i]),
                             int(lane_id[i]), float(v[i]), float(a[i]), int(lead_id[i])))

    df = pd.DataFrame(rows, columns=["run_id", "t", "vehicle_id", "x", "y", "lane_id", "v", "a", "lead_id"])
    return df


# -----------------------------
# Event & safety metrics
# -----------------------------
def compute_ttc(df: pd.DataFrame) -> pd.Series:
    """
    TTC for follower vs lead in same lane.
    TTC = gap / closing_speed when closing_speed > 0 else inf
    """
    # join lead positions at same timestamp
    lead = df[df["lead_id"] != -1][["run_id", "t", "lead_id"]].copy()
    if lead.empty:
        return pd.Series(np.inf, index=df.index)

    lead = lead.merge(
        df[["run_id", "t", "vehicle_id", "x", "v"]],
        left_on=["run_id", "t", "lead_id"],
        right_on=["run_id", "t", "vehicle_id"],
        suffixes=("", "_lead"),
        how="left",
    )

    out = pd.Series(np.inf, index=df.index, dtype=float)

    # map back to follower rows
    follower_idx = df.index[df["lead_id"] != -1]
    follower = df.loc[follower_idx]

    gap = (lead["x_lead"].values - follower["x"].values)
    closing = (follower["v"].values - lead["v_lead"].values)

    ttc = np.where((closing > 0) & (gap > 0), gap / closing, np.inf)
    out.loc[follower_idx] = ttc
    return out


def detect_events(df: pd.DataFrame, cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      events_df: per-event table
      per_step_df: original df with extra columns (ttc, near_collision, harsh_brake, disengagement)
    """
    df = df.sort_values(["run_id", "vehicle_id", "t"]).copy()

    # lane changes: lane_id changes between time steps
    df["lane_change"] = df.groupby(["run_id", "vehicle_id"])["lane_id"].diff().fillna(0).abs() > 0

    # harsh braking: acceleration below threshold
    df["harsh_brake"] = df["a"] <= cfg.harsh_brake_threshold

    # TTC and near collisions
    df["ttc"] = compute_ttc(df)
    df["near_collision"] = df["ttc"] <= cfg.ttc_threshold

    # disengagement proxy: triggered by very low TTC or extreme brake
    df["disengagement"] = (df["ttc"] <= cfg.disengagement_ttc_trigger) | (df["a"] <= cfg.disengagement_harsh_brake_trigger)

    # pullover heuristic: in rightmost lane AND slow speed sustained
    rightmost = (df["lane_id"] == cfg.road_lanes - 1)
    slow = (df["v"] <= 1.5)
    df["pullover_state"] = rightmost & slow

    # Aggregate per-vehicle/run events (count transitions rather than per-step)
    def count_transitions(s: pd.Series) -> int:
        return int((s.astype(int).diff().fillna(0) == 1).sum())

    agg = df.groupby(["run_id", "vehicle_id"]).agg(
        lane_changes=("lane_change", count_transitions),
        harsh_brake_events=("harsh_brake", count_transitions),
        near_collision_events=("near_collision", count_transitions),
        disengagement_events=("disengagement", count_transitions),
        pullover_events=("pullover_state", count_transitions),
        avg_speed=("v", "mean"),
        p95_speed=("v", lambda x: float(np.percentile(x, 95))),
        avg_ttc=("ttc", lambda x: float(np.mean(np.clip(x.replace(np.inf, np.nan), 0, 60))) if np.any(np.isfinite(x)) else np.nan),
    ).reset_index()

    # Event table in "long" format
    events = []
    for _, r in agg.iterrows():
        for name in ["lane_changes", "harsh_brake_events", "near_collision_events", "disengagement_events", "pullover_events"]:
            events.append((int(r["run_id"]), int(r["vehicle_id"]), name, int(r[name])))
    events_df = pd.DataFrame(events, columns=["run_id", "vehicle_id", "event_type", "count"])

    return events_df, df


# -----------------------------
# Fleet-level metrics
# -----------------------------
def compute_fleet_metrics(per_step_df: pd.DataFrame, agg_vehicle_df: pd.DataFrame, cfg: SimConfig) -> Dict[str, float]:
    # approximate miles driven: sum of delta x per vehicle
    per_step_df = per_step_df.sort_values(["run_id", "vehicle_id", "t"]).copy()
    per_step_df["dx"] = per_step_df.groupby(["run_id", "vehicle_id"])["x"].diff().fillna(0).clip(lower=0)
    meters = per_step_df["dx"].sum()
    miles = meters / 1609.344

    # totals
    totals = agg_vehicle_df[["lane_changes", "harsh_brake_events", "near_collision_events", "disengagement_events", "pullover_events"]].sum()

    mpd = miles / max(totals["disengagement_events"], 1)  # miles per disengagement
    per_100_miles = lambda c: (c / max(miles, 1e-6)) * 100.0

    return {
        "total_miles": float(miles),
        "miles_per_disengagement": float(mpd),
        "lane_changes_per_100_miles": float(per_100_miles(totals["lane_changes"])),
        "harsh_brakes_per_100_miles": float(per_100_miles(totals["harsh_brake_events"])),
        "near_collisions_per_100_miles": float(per_100_miles(totals["near_collision_events"])),
        "pullovers_per_100_miles": float(per_100_miles(totals["pullover_events"])),
    }


# -----------------------------
# Behavior clustering
# -----------------------------
def behavior_clustering(agg_vehicle_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Clusters vehicles by behavior profile.
    """
    features = agg_vehicle_df[["lane_changes", "harsh_brake_events", "near_collision_events", "pullover_events", "avg_speed"]].copy()
    # simple scaling
    X = (features - features.mean()) / (features.std().replace(0, 1))
    km = KMeans(n_clusters=k, random_state=7, n_init="auto")
    labels = km.fit_predict(X)
    out = agg_vehicle_df.copy()
    out["cluster"] = labels
    return out


# -----------------------------
# SQLite export + example SQL
# -----------------------------
def write_sqlite(db_path: str, per_step_df: pd.DataFrame, agg_vehicle_df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    per_step_df.to_sql("per_step_logs", conn, if_exists="replace", index=False)
    agg_vehicle_df.to_sql("vehicle_metrics", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def example_sql_queries(db_path: str) -> None:
    conn = sqlite3.connect(db_path)

    print("\nSQL: Top 10 vehicles by near-collision events")
    q1 = """
    SELECT run_id, vehicle_id, near_collision_events
    FROM vehicle_metrics
    ORDER BY near_collision_events DESC
    LIMIT 10;
    """
    print(pd.read_sql_query(q1, conn))

    print("\nSQL: Average speed by run")
    q2 = """
    SELECT run_id, AVG(avg_speed) AS avg_speed
    FROM vehicle_metrics
    GROUP BY run_id
    ORDER BY avg_speed DESC;
    """
    print(pd.read_sql_query(q2, conn))

    conn.close()


# -----------------------------
# Plots
# -----------------------------
def plot_summary(agg_vehicle_df: pd.DataFrame, clustered_df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Distribution of TTC (finite only)
    if "avg_ttc" in agg_vehicle_df.columns:
        ttc = agg_vehicle_df["avg_ttc"].dropna()
        plt.figure()
        plt.hist(ttc, bins=30)
        plt.title("Average TTC Distribution (finite only)")
        plt.xlabel("seconds")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "avg_ttc_hist.png"))
        plt.close()

    # Cluster sizes
    plt.figure()
    clustered_df["cluster"].value_counts().sort_index().plot(kind="bar")
    plt.title("Behavior Cluster Sizes")
    plt.xlabel("cluster")
    plt.ylabel("vehicles")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_sizes.png"))
    plt.close()

    # Lane changes vs harsh brakes
    plt.figure()
    plt.scatter(clustered_df["lane_changes"], clustered_df["harsh_brake_events"])
    plt.title("Lane Changes vs Harsh Braking Events")
    plt.xlabel("lane_changes")
    plt.ylabel("harsh_brake_events")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lanechange_vs_harshbrake.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = SimConfig()

    print("1) Generating synthetic fleet logs...")
    df = generate_synthetic_fleet_logs(cfg)

    print("2) Detecting events + safety metrics (TTC, harsh brakes, lane changes, pullovers, disengagements)...")
    events_df, per_step_df = detect_events(df, cfg)

    print("3) Building vehicle-level metric table...")
    # pivot event counts back into vehicle table
    agg = (events_df.pivot_table(index=["run_id", "vehicle_id"], columns="event_type", values="count", fill_value=0)
           .reset_index())
    # add additional stats from per-step
    stats = per_step_df.groupby(["run_id", "vehicle_id"]).agg(
        avg_speed=("v", "mean"),
        p95_speed=("v", lambda x: float(np.percentile(x, 95))),
        min_ttc=("ttc", lambda x: float(np.min(x)) if np.any(np.isfinite(x)) else np.inf),
    ).reset_index()

    agg_vehicle_df = agg.merge(stats, on=["run_id", "vehicle_id"], how="left")

    # rename columns to stable names
    for col in ["lane_changes", "harsh_brake_events", "near_collision_events", "disengagement_events", "pullover_events"]:
        if col not in agg_vehicle_df.columns:
            agg_vehicle_df[col] = 0

    print("4) Fleet metrics summary...")
    fleet_metrics = compute_fleet_metrics(per_step_df, agg_vehicle_df, cfg)
    for k, v in fleet_metrics.items():
        print(f"  {k}: {v:.3f}")

    print("5) Behavior clustering...")
    clustered = behavior_clustering(agg_vehicle_df, k=4)

    print("6) Writing SQLite + example SQL queries...")
    db_path = os.path.join("outputs", "fleet.db")
    write_sqlite(db_path, per_step_df, agg_vehicle_df)
    example_sql_queries(db_path)

    print("7) Creating plots...")
    plot_summary(agg_vehicle_df, clustered, out_dir=os.path.join("outputs", "figures"))

    print("\nDone ✅")
    print("Outputs:")
    print(" - outputs/fleet.db (SQLite metrics platform)")
    print(" - outputs/figures/*.png (charts)")
    print("\nInterview framing tip:")
    print("You can describe this as a simulation-based autonomy evaluation pipeline that computes fleet KPIs,")
    print("detects safety/comfort events, clusters behavior patterns, and supports SQL-based analytics.")


if __name__ == "__main__":
    main()
