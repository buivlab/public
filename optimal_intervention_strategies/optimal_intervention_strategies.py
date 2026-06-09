# import matplotlib
# matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import os
import math
import inspect
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

SEASON_MONTHS = list(range(3, 11))  # April–November
MONTHLY_ALLOCATION = [1, 1, 2, 4, 4, 6, 3, 3, 3, 2, 1, 1]
#MONTHLY_ALLOCATION = [0, 0, 0, 1, 2, 3, 5, 5, 5, 4, 2, 0]

INTERVENTION_EFFECT = [0.50, 0.69, 0.31]

TOP50 = 50
TOP100 = 100
TOP200 = 200 

# ============================================================
# LOAD CSV
# ============================================================

def load_cases_from_csv(path):

    df = pd.read_csv(path)
    df = df.sort_values(["year", "month", "fcode"])

    fcodes = df["fcode"].unique()
    fcode_to_idx = {f: i for i, f in enumerate(fcodes)}

    years = sorted(df["year"].unique())
    yearly_cases = {}

    for year in years:

        df_y = df[df["year"] == year]

        T = 12
        N = len(fcodes)
        cases = np.zeros((T, N))

        for _, row in df_y.iterrows():
            t = int(row["month"]) - 1
            n = fcode_to_idx[row["fcode"]]
            cases[t, n] = row["obs_dengue_cases"]

        yearly_cases[year] = cases

    return yearly_cases, fcodes


# ============================================================
# SIMULATION
# ============================================================

def run_simulation_explicit(cases, interventions):

    T, N = cases.shape
    cases_dyn = cases.copy()
    active = []

    interventions_by_time = {}
    for t, loc in interventions:
        interventions_by_time.setdefault(t, []).append(loc)

    total = 0

    for t in range(T):

        new_queue = []
        for loc, age, strength in active:
            if age < len(INTERVENTION_EFFECT):
                reduction = strength * INTERVENTION_EFFECT[age]
                cases_dyn[t, loc] *= (1 - reduction)
                new_queue.append((loc, age+1, strength))
        active = new_queue

        for loc in interventions_by_time.get(t, []):
            active.append((loc, 0, 1.0))

        total += cases_dyn[t].sum()

    return total


def simulate_full_trajectory(cases, interventions):

    T, N = cases.shape
    cases_dyn = cases.copy()
    active = []

    interventions_by_time = {}
    for t, loc in interventions:
        interventions_by_time.setdefault(t, []).append(loc)

    for t in range(T):

        new_queue = []
        for loc, age, strength in active:
            if age < len(INTERVENTION_EFFECT):
                reduction = strength * INTERVENTION_EFFECT[age]
                cases_dyn[t, loc] *= (1 - reduction)
                new_queue.append((loc, age+1, strength))

        active = new_queue

        for loc in interventions_by_time.get(t, []):
            active.append((loc, 0, 1.0))

    return cases_dyn


# ============================================================
# FORECAST
# ============================================================

def generate_forecast(state, t0, horizon=3, base_uncertainty=0.1):

    T, N = state.shape
    end = min(T, t0 + horizon)

    forecast = state[t0:end].copy()

    for h in range(len(forecast)):
        noise = np.random.normal(0, base_uncertainty*(h+1), size=forecast[h].shape)
        forecast[h] = np.maximum(0, forecast[h] * (1 + noise))

    return forecast

# ============================================================
# ALGORITHM HELPERS
# ============================================================

def compute_monthly_allocation(allocation_policy, budget):

    quota = np.array(allocation_policy)
    quota = quota / quota.sum()  # normalize (safety)

    raw = quota * budget
    alloc = np.floor(raw).astype(int)

    # distribute remainder
    remainder = budget - alloc.sum()
    fractional = raw - alloc

    for i in np.argsort(-fractional)[:remainder]:
        alloc[i] += 1

    return alloc  # length = 12, sums exactly to budget

# ============================================================
# GREEDY OFFLINE (UPPER BOUND)
# ============================================================

def optimise_state_aware_greedy(cases, budget, top_M=50):

    T, N = cases.shape
    interventions = []

    for _ in range(budget):

        best_gain = 0
        best_choice = None

        current_total = run_simulation_explicit(cases, interventions)
        sim_cases = simulate_full_trajectory(cases, interventions)

        for t in range(T):
            candidates = np.argsort(sim_cases[t])[-top_M:]

            for loc in candidates:
                if (t, loc) in interventions:
                    continue

                new_total = run_simulation_explicit(cases, interventions + [(t, loc)])
                gain = current_total - new_total

                if gain > best_gain:
                    best_gain = gain
                    best_choice = (t, loc)

        if best_choice is None:
            break

        interventions.append(best_choice)

    total = run_simulation_explicit(cases, interventions)
    reduction = (cases.sum() - total) / cases.sum()

    return reduction, interventions


# ============================================================
# GREEDY ONLINE
# ============================================================

def optimise_greedy_online(
    cases,
    budget,
    horizon=3,
    top_M=50,
    eps=1e-3,
    allocation_policy=None
):

    T, N = cases.shape
    interventions = []
    remaining = budget

    season_list = SEASON_MONTHS

    # ----------------------------------------
    # NEW: compute monthly allocation
    # ----------------------------------------
    if allocation_policy is not None:
        monthly_alloc = compute_monthly_allocation(allocation_policy, budget)
    else:
        monthly_alloc = None

    used_per_month = np.zeros(12, dtype=int)

    for i, t0 in enumerate(season_list):

        if remaining <= 0:
            break

        month_idx = t0 % 12

        # ----------------------------------------
        # allowed interventions this month
        # ----------------------------------------
        if monthly_alloc is not None:
            allowed_now = monthly_alloc[month_idx] - used_per_month[month_idx]
            allowed_now = min(allowed_now, remaining)
        else:
            remaining_months = len(season_list) - i
            allowed_now = max(1, remaining // remaining_months)

        if allowed_now <= 0:
            continue

        current_state = simulate_full_trajectory(cases, interventions)
        forecast = generate_forecast(current_state, t0, horizon)

        if len(forecast) == 0:
            continue

        used = 0

        while remaining > 0 and used < allowed_now:

            best_gain = 0
            best_loc = None

            candidates = np.argsort(forecast[0])[-top_M:]

            for loc in candidates:

                # skip duplicate
                if (t0, loc) in interventions:
                    continue

                # diminishing returns
                if any((abs(tt - t0) <= 2 and l == loc) for (tt, l) in interventions):
                    continue

                base = forecast[0].sum()

                new = forecast[0].copy()
                new[loc] *= (1 - INTERVENTION_EFFECT[0])

                gain = base - new.sum()

                if gain > best_gain:
                    best_gain = gain
                    best_loc = loc

            if best_gain < eps or best_loc is None:
                break

            interventions.append((t0, best_loc))
            remaining -= 1
            used += 1
            used_per_month[month_idx] += 1

    total = run_simulation_explicit(cases, interventions)
    reduction = (cases.sum() - total) / cases.sum()

    return reduction, interventions

# ============================================================
# HISTORICAL CUMMULATIVE RANKING ALLOCATION
# ============================================================

def optimise_historical_cummulative_ranking(
    cases,
    budget,
    allocation_policy=None
):

    T, N = cases.shape

    # ----------------------------------------
    # rank communes by total burden
    # ----------------------------------------
    total_cases_per_commune = cases.sum(axis=0)
    ranked_communes = np.argsort(total_cases_per_commune)[::-1]

    interventions = []

    # ----------------------------------------
    # NEW: compute monthly allocation
    # ----------------------------------------
    if allocation_policy is not None:
        monthly_alloc = compute_monthly_allocation(allocation_policy, budget)
    else:
        monthly_alloc = None

    used_per_month = np.zeros(12, dtype=int)

    idx = 0  # pointer over ranked communes

    for t0 in SEASON_MONTHS:

        if len(interventions) >= budget:
            break

        month_idx = t0 % 12

        # ----------------------------------------
        # allowed interventions this month
        # ----------------------------------------
        if monthly_alloc is not None:
            allowed_now = monthly_alloc[month_idx] - used_per_month[month_idx]
        else:
            allowed_now = 1  # fallback (same as original sequential logic)

        if allowed_now <= 0:
            continue

        used = 0

        while len(interventions) < budget and used < allowed_now:

            loc = ranked_communes[idx % N]

            # (optional) avoid duplicate (t0, loc)
            if (t0, loc) not in interventions:
                interventions.append((t0, loc))
                used += 1
                used_per_month[month_idx] += 1

            idx += 1

    # ----------------------------------------
    # evaluation
    # ----------------------------------------
    total = run_simulation_explicit(cases, interventions)
    baseline = cases.sum()

    reduction = (baseline - total) / baseline

    return reduction, interventions

# ============================================================
# HISTORICAL MONTHLY RANKING ALLOCATION
# ============================================================

def optimise_historical_monthly_ranking(
    cases,
    budget,
    hist_cases,    
    allocation_policy=None,    
):
    """
    cases: (T, N)
    hist_cases: (Y, T, N) or list of (T, N)
    budget: total interventions
    allocation_policy: length-12 proportions
    """

    T, N = cases.shape

    # ----------------------------------------
    # 1. stack historical data
    # ----------------------------------------
    if isinstance(hist_cases, list):
        hist_array = np.stack(hist_cases)
    else:
        hist_array = hist_cases

    # ----------------------------------------
    # 2. compute monthly averages
    # ----------------------------------------
    monthly_scores = hist_array.mean(axis=0)  # (T, N)

    # ----------------------------------------
    # 3. ranking per time step
    # ----------------------------------------
    monthly_rankings = {}
    for t in range(T):
        monthly_rankings[t] = np.argsort(monthly_scores[t])[::-1]

    # ----------------------------------------
    # 4. compute monthly allocation
    # ----------------------------------------
    if allocation_policy is not None:
        monthly_alloc = compute_monthly_allocation(allocation_policy, budget)
    else:
        monthly_alloc = None

    used_per_month = np.zeros(12, dtype=int)

    # track ranking pointer per time step
    month_counters = {t: 0 for t in SEASON_MONTHS}

    interventions = []

    # ----------------------------------------
    # 5. allocate interventions over time
    # ----------------------------------------
    for t0 in SEASON_MONTHS:

        if len(interventions) >= budget:
            break

        month_idx = t0 % 12

        # allowed interventions this month
        if monthly_alloc is not None:
            allowed_now = monthly_alloc[month_idx] - used_per_month[month_idx]
        else:
            allowed_now = 1  # fallback (original behavior)

        if allowed_now <= 0:
            continue

        used = 0

        while len(interventions) < budget and used < allowed_now:

            rank_list = monthly_rankings[t0]

            idx = month_counters[t0] % N
            loc = rank_list[idx]

            # optional: avoid duplicate (t0, loc)
            if (t0, loc) not in interventions:
                interventions.append((t0, loc))
                used += 1
                used_per_month[month_idx] += 1

            month_counters[t0] += 1

    # ----------------------------------------
    # 6. evaluation
    # ----------------------------------------
    total = run_simulation_explicit(cases, interventions)
    baseline = cases.sum()

    reduction = (baseline - total) / baseline

    return reduction, interventions

# ============================================================
# MPC ONLINE
# ============================================================

def optimise_mpc(
    cases,
    budget,
    horizon=3,
    top_M=50,
    eps=1e-3,
    allocation_policy=None
):

    T, N = cases.shape
    interventions = []
    remaining = budget

    season_list = SEASON_MONTHS

    # ----------------------------------------
    # NEW: precompute monthly allocation
    # ----------------------------------------
    if allocation_policy is not None:
        monthly_alloc = compute_monthly_allocation(allocation_policy, budget)
    else:
        monthly_alloc = None

    # track usage per month
    used_per_month = np.zeros(12, dtype=int)

    for i, t0 in enumerate(season_list):

        if remaining <= 0:
            break

        month_idx = t0 % 12

        # ----------------------------------------
        # allowed interventions this month
        # ----------------------------------------
        if monthly_alloc is not None:
            allowed_now = monthly_alloc[month_idx] - used_per_month[month_idx]
            allowed_now = min(allowed_now, remaining)
        else:
            remaining_months = len(season_list) - i
            allowed_now = max(1, remaining // remaining_months)

        if allowed_now <= 0:
            continue

        current_state = simulate_full_trajectory(cases, interventions)
        forecast = generate_forecast(current_state, t0, horizon)

        H = forecast.shape[0]
        if H == 0:
            continue

        used = 0

        while remaining > 0 and used < allowed_now:

            best_gain = 0
            best_loc = None

            candidates = np.argsort(forecast[0])[-top_M:]

            for loc in candidates:

                if (t0, loc) in interventions:
                    continue

                # diminishing returns
                if any((abs(tt - t0) <= 2 and l == loc) for (tt, l) in interventions):
                    continue

                # rollout plan
                plan = [(0, loc)]
                for h in range(1, H):
                    future_loc = np.argmax(forecast[h])
                    plan.append((h, future_loc))

                base = run_simulation_explicit(forecast, [])
                new  = run_simulation_explicit(forecast, plan)

                gain = base - new

                if gain > best_gain:
                    best_gain = gain
                    best_loc = loc

            if best_gain < eps or best_loc is None:
                break

            interventions.append((t0, best_loc))
            remaining -= 1
            used += 1
            used_per_month[month_idx] += 1

    total = run_simulation_explicit(cases, interventions)
    reduction = (cases.sum() - total) / cases.sum()

    return reduction, interventions

# ============================================================
# RANDOM SEASONAL
# ============================================================

def optimise_random_seasonal(cases, budget, seed=42):

    np.random.seed(seed)

    T, N = cases.shape

    valid_pairs = [(t, i) for t in SEASON_MONTHS for i in range(N)]

    selected = np.random.choice(
        len(valid_pairs),
        size=min(budget, len(valid_pairs)),
        replace=False
    )

    interventions = [valid_pairs[i] for i in selected]

    total = run_simulation_explicit(cases, interventions)
    reduction = (cases.sum() - total) / cases.sum()

    return reduction, interventions


# ============================================================
# TRACKING + ANALYSIS
# ============================================================

def analyze_interventions(interventions, T, N, fcodes):

    df = pd.DataFrame(interventions, columns=["month", "commune"])

    # Monthly counts
    monthly_counts = df["month"].value_counts().sort_index()

    # Commune frequency
    commune_counts = df["commune"].value_counts()

    # Heatmap matrix
    heatmap = np.zeros((T, N))
    for t, loc in interventions:
        heatmap[t, loc] += 1

    print("\nTop 10 Communes:")
    top = commune_counts.head(10)
    for idx, count in top.items():
        print(f"{fcodes[idx]}: {count}")

    return df, monthly_counts, commune_counts, heatmap

def prepare_commune_map_data(commune_counts, fcodes):
    """
    Convert commune counts into a dataframe for choropleth plotting
    """
    df_map = pd.DataFrame({
        "l2_code": [fcodes[idx] for idx in commune_counts.index],
        "interventions": commune_counts.values
    })

    return df_map

def prepare_full_map_data(commune_counts, fcodes, geojson):

    # --- Step 1: build your working df (THIS part was correct)
    df_counts = pd.DataFrame({
        "l2_code": [fcodes[idx] for idx in commune_counts.index],
        "interventions": commune_counts.values
    })

    # --- Step 2: extract ALL communes from geojson
    all_codes = extract_all_communes(geojson)
    df_all = pd.DataFrame({"l2_code": all_codes})

    # --- Step 3: enforce consistent type (critical)
    df_counts["l2_code"] = df_counts["l2_code"].astype(str).str.strip()
    df_all["l2_code"] = df_all["l2_code"].astype(str).str.strip()

    # --- Step 4: merge instead of map (THIS is the key fix)
    df_map = df_all.merge(df_counts, on="l2_code", how="left")

    # --- Step 5: fill missing communes
    df_map["interventions"] = df_map["interventions"].fillna(0)

    return df_map

def extract_all_communes(geojson):
    return [feature["properties"]["l2_code"] for feature in geojson["features"]]

# ============================================================
# VISUALIZATION
# ============================================================

def plot_intervention_analysis(monthly_counts, heatmap, fcodes, title):

    # Ensure all months exist
    monthly_counts = monthly_counts.reindex(range(12), fill_value=0)

    plt.figure(figsize=(12,4))

    # Monthly interventions
    plt.subplot(1,2,1)        
    labels = [str(i+1) for i in monthly_counts.index]
    plt.bar(labels, monthly_counts.values)
    plt.title("Interventions per Month")
    plt.xlabel("Month")
    plt.ylabel("Count")

    # Heatmap
    month_labels = [str(i+1) for i in range(heatmap.shape[0])]    
    plt.subplot(1,2,2)
    sns.heatmap(heatmap, cmap="Reds")
    plt.title("Commune Intervention Heatmap")
    plt.xlabel("Commune Index")
    plt.ylabel("Month")
    plt.yticks(
        ticks=np.arange(heatmap.shape[0]) + 0.5,
        labels=month_labels,
        rotation=0
    )

    plt.ylabel("Month")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_and_save_intervention_analysis(monthly_counts, heatmap, fcodes, title, save_path):

    # Ensure all months exist
    monthly_counts = monthly_counts.reindex(range(12), fill_value=0)

    plt.figure(figsize=(12,4))

    # Monthly interventions
    plt.subplot(1,2,1)
    labels = [str(i+1) for i in monthly_counts.index]
    plt.bar(labels, monthly_counts.values)
    plt.title("Interventions per Month")
    plt.xlabel("Month")
    plt.ylabel("Count")

    # Heatmap
    # month_labels = [str(i+1) for i in range(heatmap.shape[0])]    
    # plt.subplot(1,2,2)
    # sns.heatmap(heatmap, cmap="Reds")
    # plt.title("Commune Intervention Heatmap")
    # plt.xlabel("Commune Index")
    # plt.ylabel("Month")
    # plt.yticks(
    #     ticks=np.arange(heatmap.shape[0]) + 0.5,
    #     labels=month_labels,
    #     rotation=0
    # )

    # Heatmap (axes flipped)
    month_labels = [str(i + 1) for i in range(heatmap.shape[0])]
    plt.subplot(1, 2, 2)
    # transpose matrix so axes swap
    sns.heatmap(heatmap.T, cmap="Reds")
  
    plt.title("Commune Intervention Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Commune Index")

    # month labels now belong on x-axis
    plt.xticks(
        ticks=np.arange(heatmap.shape[0]) + 0.5,
        labels=month_labels,
        rotation=0
    )

    plt.suptitle(title)
    plt.tight_layout()

    # Save instead of show
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # VERY important (prevents memory issues)

def plot_budget_curve(df):

    plt.figure(figsize=(9,5))

    plt.plot(df["budget"], df["greedy"], marker="o", label="Greedy Offline")
    plt.plot(df["budget"], df["greedy_online"], marker="o", label="Greedy Online")
    plt.plot(df["budget"], df["mpc"], marker="o", label="MPC")

    plt.plot(df["budget"], df["historical_cummulative"], marker="o", linestyle="--", label="Historical (Cummulative)")
    plt.plot(df["budget"], df["historical_monthly"], marker="o", linestyle="--", label="Historical (Monthly)")
    plt.plot(df["budget"], df["random_seasonal"], marker="o", linestyle="--", label="Random (Seasonal)")

    plt.xlabel("Intervention Number (Budget)")
    plt.ylabel("Reduction (%)")
    plt.title("Policy Comparison: Budget vs Reduction")

    plt.legend()
    plt.grid()

    plt.show()


def plot_intervention_map(df_map, geojson, title, max_color=None):

    if max_color is None:
        max_color = df_map["interventions"].max()

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="l2_code",
        featureidkey="properties.l2_code",
        color="interventions",
        color_continuous_scale=["lightblue", "orange", "red"],
        range_color=(0, max_color),
        hover_data={"l2_code": True, "interventions": True},
    )

    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
        title=title,
        coloraxis_colorbar=dict(
            title="Interventions",
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            len=0.7,
            thickness=12
        ),
        margin=dict(l=0, r=0, t=50, b=60)
    )

    return fig

def plot_intervention_map_full(df_map, geojson, title):

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="l2_code",
        featureidkey="properties.l2_code",
        color="interventions",
        color_continuous_scale=[
            [0.0, "white"],     # no intervention
            [0.01, "lightblue"],
            [0.5, "orange"],
            [1.0, "red"]
        ],
        #range_color=(0, df_map["interventions"].max()),
        range_color=(0, 10),
        hover_data={"l2_code": True, "interventions": True},
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=True
    )

    # makes borders visible
    fig.update_traces(
        marker_line_color="black",
        marker_line_width=0.3
    )

    fig.update_layout(
        title=title,
        coloraxis_colorbar=dict(
            title="Interventions",
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            len=0.7,
            thickness=12
        ),
        margin=dict(l=0, r=0, t=50, b=60)
    )

    return fig

# ============================================================
# 9. BUDGET CURVE
# ============================================================

def compute_budget_curve_all(cases, budgets, top_M=50, hist_cases=None, allocation_policy=None):

    results = []

    for b in budgets:
        print(f"\n===== Budget {b} =====")
        
        r_mpc, _ = optimise_mpc(cases, b, top_M=top_M, allocation_policy=allocation_policy)
        # r_mpc, _ = optimise_mpc(cases, b, top_M=top_M, allocation_policy=None)
        r_greedy_online, _ =  optimise_greedy_online(cases, b, top_M=top_M, allocation_policy=allocation_policy)

        # bench marking        
        r_greedy, _ = optimise_state_aware_greedy(cases, b, top_M)        
        r_hist, _ = optimise_historical_cummulative_ranking(cases, b, allocation_policy=allocation_policy)
        r_hist_month, _ = optimise_historical_monthly_ranking(cases, b, hist_cases, allocation_policy=allocation_policy)
        r_random, _ = optimise_random_seasonal(cases, b)

        results.append({            
            "budget": b,
            "mpc": r_mpc,            
            "greedy": r_greedy,
            "greedy_online": r_greedy_online,
            "random_seasonal": r_random,
            "historical_cummulative": r_hist,
            "historical_monthly": r_hist_month
        })

    return pd.DataFrame(results)


def evaluate_strategies(
    year,
    cases,
    fcodes,
    geojson,
    hist_cases,
    budget,
    top_M,
    output_dir="output"
):

    print(f"\n================ YEAR {year} ================\n")
    # -------------------------
    # Helper to reduce repetition
    # -------------------------
    def run_and_plot(name, optimise_fn, *args, save_tag="", plot_map=True, **kwargs):

        print(f"\n=== {name} ===")

        result, interventions = optimise_fn(*args, **kwargs)

        df, m, cc, h = analyze_interventions(
            interventions, 12, cases.shape[1], fcodes
        )

        # Save matplotlib figure
        plot_and_save_intervention_analysis(
            m, h, fcodes, name,
            save_path=f"{output_dir}/{year}_{save_tag}_analysis.png"
        )

        if plot_map:
            df_map_full = prepare_full_map_data(cc, fcodes, geojson)

            fig_map = plot_intervention_map_full(
                df_map_full,
                geojson,
                title=f"{name} ({year})"
            )

            fig_map.write_image(
                f"{output_dir}/{year}_{save_tag}_map.png", scale=2
            )

        return result, interventions

    # =========================
    # RUN STRATEGIES
    # =========================

    run_and_plot(
        "Greedy Offline",
        optimise_state_aware_greedy,
        cases, budget,
        top_M=top_M,
        save_tag="greedy_offline"
    )

    run_and_plot(
        "Greedy Online",
        optimise_greedy_online,
        cases, budget,
        top_M=top_M,
        allocation_policy=MONTHLY_ALLOCATION,
        save_tag="greedy_online"
    )

    run_and_plot(
        "MPC",
        optimise_mpc,
        cases, budget,
        top_M=top_M,
        allocation_policy=MONTHLY_ALLOCATION,
        save_tag="mpc"
    )

    run_and_plot(
        "Random",
        optimise_random_seasonal,
        cases, budget,
        save_tag="random",
        plot_map=False  # optional (usually not needed)
    )

    run_and_plot(
        "Historical Monthly Ranking",
        optimise_historical_monthly_ranking,
        cases, budget, hist_cases, 
        allocation_policy = MONTHLY_ALLOCATION,
        save_tag="hist_monthly"
    )

    run_and_plot(
        "Historical Cumulative Ranking",
        optimise_historical_cummulative_ranking,
        cases, budget,
        allocation_policy = MONTHLY_ALLOCATION,
        save_tag="hist_cumulative"
    )


def plot_intervention_maps(
    strategy_name,
    optimise_fn,
    years,
    yearly_cases,
    fcodes,
    geojson,
    budget,
    hist_cases=None,
    top_M=None,
    allocation_policy=None,
    cols=3,
    output_path="output/strategy_comparison.png"
):
    """
    Plot intervention maps for a given strategy across multiple years.

    cols: number of subplots per row (2 or 3 recommended)
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image_paths = []

    # -------------------------
    # Generate maps for each year
    # -------------------------
    for YEAR in years:

        if YEAR not in yearly_cases:
            print(f"Skipping {YEAR}")
            continue

        print(f"{strategy_name} - {YEAR}")

        cases = yearly_cases[YEAR]

        # ---- Run optimisation (handle kwargs safely)
        kwargs = {}
        if top_M is not None:
            kwargs["top_M"] = top_M
        if hist_cases is not None:
            kwargs["hist_cases"] = hist_cases
        if allocation_policy is not None:
            kwargs["allocation_policy"] = allocation_policy

        result, interventions = optimise_fn(cases, budget, **kwargs)

        # ---- Analyze
        df, m, cc, h = analyze_interventions(
            interventions, 12, cases.shape[1], fcodes
        )

        # ---- Prepare map
        df_map = prepare_full_map_data(cc, fcodes, geojson)

        fig_map = plot_intervention_map_full(
            df_map,
            geojson,
            title=f"{YEAR}"
        )

        # Save temporary image
        img_path = f"output/tmp_{strategy_name}_{YEAR}.png"
        fig_map.write_image(img_path, scale=2)
        image_paths.append((YEAR, img_path))

    # -------------------------
    # Create grid figure
    # -------------------------
    n = len(image_paths)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

    # Flatten axes for easy indexing
    axes = axes.flatten() if n > 1 else [axes]

    for i, (YEAR, path) in enumerate(image_paths):

        img = Image.open(path)
        axes[i].imshow(img)
        # axes[i].set_title(f"{YEAR}")
        axes[i].axis("off")

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(strategy_name, fontsize=16)
    plt.tight_layout()

    # Save final figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_efficiency_across_years(
    years,
    yearly_cases,
    strategies,   # dict: {"Greedy": fn, "MPC": fn, ...}
    budget,
    top_M=None,
    hist_cases=None,
    smooth=False,
    window=3,
    output_path="output/efficiency_across_years.png"
):
    """
    Plot efficiency (cases averted per intervention) across years
    for multiple strategies in a single figure.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    efficiency_results = {name: [] for name in strategies}
    valid_years = []

    # -------------------------
    # Loop through years
    # -------------------------
    for YEAR in years:

        if YEAR not in yearly_cases:
            print(f"Skipping {YEAR}")
            continue

        print(f"Processing {YEAR}")
        cases = yearly_cases[YEAR]
        baseline_total = cases.sum()

        valid_years.append(YEAR)

        # -------------------------
        # Evaluate each strategy
        # -------------------------
        for name, optimise_fn in strategies.items():

            # ---- build candidate kwargs ----
            candidate_kwargs = {
                "top_M": top_M,
                "hist_cases": hist_cases
            }

            # ---- filter based on function signature ----
            sig = inspect.signature(optimise_fn)
            valid_kwargs = {
                k: v for k, v in candidate_kwargs.items()
                if k in sig.parameters and v is not None
            }

            # ---- run optimisation ----
            result, interventions = optimise_fn(cases, budget, **valid_kwargs)

            # ---- simulate ----
            sim_cases = simulate_full_trajectory(cases, interventions)

            # ---- compute efficiency ----
            total_cases = sim_cases.sum()
            averted = baseline_total - total_cases
            efficiency = averted / budget if budget > 0 else 0.0

            efficiency_results[name].append(efficiency)

    # -------------------------
    # Optional smoothing
    # -------------------------
    if smooth:
        for name in efficiency_results:
            values = efficiency_results[name]
            if len(values) >= window:
                efficiency_results[name] = np.convolve(
                    values,
                    np.ones(window)/window,
                    mode='same'
                )

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(8,5))

    for name, values in efficiency_results.items():
        plt.plot(valid_years, values, marker='o', label=name)

    plt.xlabel("Year")
    plt.ylabel("Cases Averted per Intervention")
    plt.title(f"Efficiency Across Years (Budget = {budget})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

    return efficiency_results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    GEOJSON_PATH = "geodata/2025/geoboundary/mdr_admin_boundary_level2_2025.geojson"
    HIST_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  # manually chosen
    PLOT_YEARS = [2020,2021,2022, 2023, 2024, 2025]
    PLOT_YEARS = [2018,2022]

    # Load geojson
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    yearly_cases, fcodes = load_cases_from_csv(
        "edengue/data/model_input_data/model_input_data_mdr_lev2_2026.csv"
    )

    hist_cases = [yearly_cases[y] for y in HIST_YEARS]

    YEARS = list(range(2020, 2026))  # full range
           
    # =========================
    # BULK RUN
    # =========================
    # Run for all years

    for year in PLOT_YEARS:
        if year not in yearly_cases:
            print(f"Skipping {year} (no data)")
            continue
        cases = yearly_cases[year]    

        evaluate_strategies(
            year,
            cases,
            fcodes,
            geojson,
            hist_cases,
            budget=200,
            top_M=TOP100
        )


    # plot_intervention_maps(
    #     strategy_name="Greedy Offline Strategy",
    #     optimise_fn=optimise_state_aware_greedy,
    #     years=PLOT_YEARS,
    #     yearly_cases=yearly_cases,
    #     fcodes=fcodes,
    #     geojson=geojson,
    #     budget=200,
    #     # top_M=TOP100,
    #     # allocation_policy=MONTHLY_ALLOCATION,
    #     cols=2,
    #     output_path="output/greedy_allocation_map.png"
    # )

    # plot_intervention_maps(
    #     strategy_name="Greedy Online Strategy",
    #     optimise_fn=optimise_greedy_online,
    #     years=PLOT_YEARS,
    #     yearly_cases=yearly_cases,
    #     fcodes=fcodes,
    #     geojson=geojson,
    #     budget=200,
    #     top_M=TOP100,
    #     allocation_policy=MONTHLY_ALLOCATION,
    #     cols=2,
    #     output_path="output/greedy_online_allocation_map.png"
    # )

    # plot_intervention_maps(
    #     strategy_name="Historical Cummulative Ranking Strategy",
    #     optimise_fn=optimise_historical_cummulative_ranking,
    #     years=PLOT_YEARS,
    #     yearly_cases=yearly_cases,
    #     fcodes=fcodes,
    #     geojson=geojson,
    #     budget=200,
    #     allocation_policy=MONTHLY_ALLOCATION,
    #     cols=2,
    #     output_path="output/hist_cummulative_ranking_allocation_map.png"
    # )

    # plot_intervention_maps(
    #     strategy_name="Historical Monthly Ranking Strategy",
    #     optimise_fn=optimise_historical_monthly_ranking,
    #     years=PLOT_YEARS,
    #     yearly_cases=yearly_cases,
    #     fcodes=fcodes,
    #     geojson=geojson,
    #     budget=200,
    #     hist_cases=hist_cases,
    #     allocation_policy=MONTHLY_ALLOCATION,
    #     cols=2,
    #     output_path="output/hist_monthly_ranking_allocation_map.png"
    # )

    # plot_intervention_maps(
    #     strategy_name="Model Predictive Control Strategy",
    #     optimise_fn=optimise_mpc,
    #     years=PLOT_YEARS,
    #     yearly_cases=yearly_cases,
    #     fcodes=fcodes,
    #     geojson=geojson,
    #     budget=200,
    #     top_M=TOP100,
    #     allocation_policy=MONTHLY_ALLOCATION,
    #     cols=2,
    #     output_path="output/mpc_allocation_map.png"
    # )


    # =========================
    # SINGLE RUN
    # =========================
    budget = 200
    top_M = TOP100
    YEAR = 2022
    cases = yearly_cases[YEAR]    

    # print("\n=== GREEDY OFFLINE ===")
    # r1, int1 = optimise_state_aware_greedy(cases, budget, top_M=top_M)
    # df1, m1, cc1, h1 = analyze_interventions(int1, 12, cases.shape[1], fcodes)
    # plot_intervention_analysis(m1, h1, fcodes, "Greedy Offline")
    
    # # Plot map
    # df_map_full = prepare_full_map_data(cc1, fcodes, geojson)
    # fig_map = plot_intervention_map_full(
    #     df_map_full,
    #     geojson,
    #     title="Greedy Offline Strategy"
    # )
    # fig_map.write_image(f"output/{YEAR}_greedy_intervention_map_.png", scale=3)

    # # Plot map
    # # df_map = prepare_commune_map_data(cc1, fcodes)
    # # fig_map = plot_intervention_map(
    # #     df_map,
    # #     geojson,
    # #     title="Spatial Distribution of Interventions"
    # # )
    # # fig_map.write_image("output/2022_greedy_intervention_map.png", scale=2)

    # print("\n=== GREEDY ONLINE ===")
    # r2, int2 = optimise_greedy_online(cases, budget, top_M=top_M, allocation_policy=MONTHLY_ALLOCATION)
    # df2, m2, cc2, h2 = analyze_interventions(int2, 12, cases.shape[1], fcodes)
    # plot_intervention_analysis(m2, h2, fcodes, "Greedy Online")
    # # Plot map
    # df_map_full = prepare_full_map_data(cc2, fcodes, geojson)
    # fig_map = plot_intervention_map_full(
    #     df_map_full,
    #     geojson,
    #     title="Greedy Online Strategy"
    # )
    # fig_map.write_image(f"output/{YEAR}_greedy_online_intervention_map_.png", scale=3)


    # print("\n=== MPC ===")
    # r3, int3 = optimise_mpc(cases, budget, top_M=top_M, allocation_policy=MONTHLY_ALLOCATION)
    # df3, m3, cc3, h3 = analyze_interventions(int3, 12, cases.shape[1], fcodes)
    # plot_intervention_analysis(m3, h3, fcodes, "MPC")
    # # Plot map
    # df_map_full = prepare_full_map_data(cc3, fcodes, geojson)
    # fig_map = plot_intervention_map_full(
    #     df_map_full,
    #     geojson,
    #     title="Model Predictive Control Strategy"
    # )
    # fig_map.write_image(f"output/{YEAR}_mpc_intervention_map_.png", scale=3)


    # print("\n=== RANDOM ===")
    # r4, int4 = optimise_random_seasonal(cases, budget)
    # df4, m4, _, h4 = analyze_interventions(int4, 12, cases.shape[1], fcodes)
    # plot_intervention_analysis(m4, h4, fcodes, "Random")

    # print("\n=== HISTORICAL MONTHLY (SEASONAL) ===")
    # r5, int5 = optimise_historical_monthly_ranking(cases, hist_cases, budget, allocation_policy=MONTHLY_ALLOCATION)
    # df5, m5, cc5, h5 = analyze_interventions(int5, 12, cases.shape[1], fcodes)
    # plot_intervention_analysis(m5, h5, fcodes, "Historical Monthly Ranking")
    # # Plot map
    # df_map_full = prepare_full_map_data(cc5, fcodes, geojson)
    # fig_map = plot_intervention_map_full(
    #     df_map_full,
    #     geojson,
    #     title="Historical Monthly Ranking Strategy"
    # )
    # fig_map.write_image(f"output/{YEAR}_histmon_intervention_map_.png", scale=3)

    # print("\n=== HISTORICAL (SEASONAL) ===")
    # r6, int6 = optimise_historical_cummulative_ranking(cases, budget, allocation_policy=MONTHLY_ALLOCATION)
    # df6, m6, cc6, h6 = analyze_interventions(int6, 12, cases.shape[1], fcodes)
    # plot_intervention_analysis(m6, h6, fcodes, "Historical Cummulative Ranking")
    # # Plot map
    # df_map_full = prepare_full_map_data(cc6, fcodes, geojson)
    # fig_map = plot_intervention_map_full(
    #     df_map_full,
    #     geojson,
    #     title="Historical Cummulative Ranking Strategy"
    # )
    # fig_map.write_image(f"output/{YEAR}_histcumm_intervention_map_.png", scale=3)

    # =========================
    # BUDGET CURVE
    # =========================
    # budgets = [0, 10, 20, 50, 100, 200]

    # df_curve = compute_budget_curve_all(cases, budgets, hist_cases=hist_cases, top_M=TOP100, allocation_policy=MONTHLY_ALLOCATION)

    # print("\nPolicy Curve:\n", df_curve)

    # plot_budget_curve(df_curve)

    # =========================
    # EFFICIENCY CURVE
    # =========================
    # strategies = {
    #     "Greedy (Offline)": optimise_state_aware_greedy,
    #     "Greedy (Online)": optimise_greedy_online,        
    #     "MPC": optimise_mpc,
    #     "Historical Cummulative Ranking": optimise_historical_cummulative_ranking,
    #     "Historical Monthly Ranking": optimise_historical_monthly_ranking,
    #     "Random Allocation": optimise_random_seasonal,
    # }

    # eff_results = plot_efficiency_across_years(
    #     years=list(range(2010, 2026)),
    #     yearly_cases=yearly_cases,
    #     strategies=strategies,
    #     hist_cases=hist_cases,
    #     budget=200,
    #     top_M=TOP100,
    #     smooth=True
    # )    