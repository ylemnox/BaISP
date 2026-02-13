import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils import ensure_dir, dataframe_to_float


NOISE_MODELS = {
    "gaussian": stats.norm,
    "log-normal": stats.lognorm,
    "uniform": stats.uniform,
    "exponential": stats.expon,
    "gamma": stats.gamma,
}


def _fit_model(data: np.ndarray, model_name: str, model_func) -> Dict[str, Any]:
    clean = data[np.isfinite(data)]
    if len(clean) < 3:
        return {"ok": False}

    if model_name in ["log-normal", "exponential", "gamma"]:
        clean = clean[clean > 0]
        if len(clean) < 3:
            return {"ok": False}

    try:
        if model_name in ["log-normal", "exponential", "gamma"]:
            params = model_func.fit(clean, floc=0)
        else:
            params = model_func.fit(clean)

        ks_stat, p_value = stats.kstest(clean, lambda x: model_func.cdf(x, *params))
        return {
            "ok": True,
            "params": params,
            "p_value": float(p_value),
            "n": len(clean),
        }
    except Exception:
        return {"ok": False}


def _row_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if "Spike Time" not in df.columns:
        raise ValueError("Expected 'Spike Time' column in input CSV")
    value_cols = [c for c in df.columns if c != "Spike Time"]
    values = df[value_cols].values.astype(float)
    return values, value_cols


def analyze_noise_models(df: pd.DataFrame) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    values, _ = _row_data(df)
    results: Dict[int, Dict[str, Any]] = {}
    model_names = list(NOISE_MODELS.keys())

    for row_idx in range(values.shape[0]):
        row_values = values[row_idx]
        results[row_idx] = {}
        for name in model_names:
            res = _fit_model(row_values, name, NOISE_MODELS[name])
            results[row_idx][name] = res

    return results, model_names


def _top_models_for_row(row_results: Dict[str, Any], top_n: int = 3) -> List[Tuple[str, float]]:
    items = []
    for name, res in row_results.items():
        if res.get("ok"):
            items.append((name, res["p_value"]))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]


def select_models_interactive(df: pd.DataFrame, results: Dict[int, Dict[str, Any]], model_names: List[str]) -> Dict[int, str]:
    print("\nTop 3 models per row (by p-value):")
    print("-" * 80)

    spike_times = df["Spike Time"].values
    for row_idx in range(len(spike_times)):
        top3 = _top_models_for_row(results[row_idx])
        display = ", ".join([f"{name} (p={pval:.4f})" for name, pval in top3])
        print(f"Row {row_idx + 1} | Spike Time {spike_times[row_idx]:.6f} | {display}")

    print("\nSelect model mode:")
    print("1. Same model for all rows")
    print("2. Select per row")

    while True:
        choice = input("Enter choice (1-2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter 1 or 2")

    selected: Dict[int, str] = {}

    if choice == "1":
        avg_scores = {}
        for name in model_names:
            pvals = [results[i][name]["p_value"] for i in results if results[i][name].get("ok")]
            avg_scores[name] = np.mean(pvals) if pvals else -1.0
        best = max(avg_scores.items(), key=lambda x: x[1])[0]
        print("\nAverage p-values by model:")
        for name, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= 0:
                print(f"  {name}: {score:.4f}")
        print(f"\nSelected model for all rows: {best}")
        for row_idx in results:
            selected[row_idx] = best
    else:
        for row_idx in results:
            top3 = _top_models_for_row(results[row_idx])
            default = top3[0][0] if top3 else model_names[0]
            prompt = f"Row {row_idx + 1} select model [{default}]: "
            user = input(prompt).strip().lower()
            if not user:
                selected[row_idx] = default
            elif user in model_names:
                selected[row_idx] = user
            else:
                print(f"Unknown model '{user}', using default {default}")
                selected[row_idx] = default

    return selected


def _param_for_model(model_name: str, params: Tuple[float, ...]) -> float:
    if model_name == "gaussian":
        return float(params[1])  # sigma
    if model_name == "log-normal":
        return float(params[0])  # shape (sigma)
    if model_name == "gamma":
        return float(params[0])  # shape
    if model_name == "exponential":
        return float(params[1])  # scale
    if model_name == "uniform":
        return float(params[1])  # width
    return float("nan")


def fit_selected_models(
    df: pd.DataFrame,
    results: Dict[int, Dict[str, Any]],
    selected_models: Dict[int, str],
    output_dir: str,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    rows = []

    spike_times = df["Spike Time"].values

    for row_idx in range(len(spike_times)):
        model = selected_models[row_idx]
        res = results[row_idx].get(model, {})
        if not res.get("ok"):
            rows.append({
                "row_index": row_idx,
                "spike_time": float(spike_times[row_idx]),
                "model": model,
                "p_value": float("nan"),
                "param": float("nan"),
                "params_json": "{}",
            })
            continue

        param = _param_for_model(model, res["params"])
        rows.append({
            "row_index": row_idx,
            "spike_time": float(spike_times[row_idx]),
            "model": model,
            "p_value": res["p_value"],
            "param": param,
            "params_json": json.dumps([float(x) for x in res["params"]]),
        })

    result_df = pd.DataFrame(rows)
    return result_df


def plot_noise_fits(df: pd.DataFrame, results_df: pd.DataFrame, output_dir: str, filename_prefix: str = "") -> None:
    ensure_dir(output_dir)
    values, _ = _row_data(df)

    for row_idx in range(values.shape[0]):
        row_values = values[row_idx]
        row_values = row_values[np.isfinite(row_values)]
        if len(row_values) == 0:
            continue

        row_meta = results_df[results_df["row_index"] == row_idx]
        if row_meta.empty:
            continue

        model_name = row_meta.iloc[0]["model"]
        params = json.loads(row_meta.iloc[0]["params_json"]) if row_meta.iloc[0]["params_json"] else []

        plt.figure(figsize=(6, 4))
        plt.hist(row_values, bins=max(10, min(30, len(row_values) // 2)), density=True, alpha=0.7, color="lightblue", edgecolor="black")

        if model_name in NOISE_MODELS and params:
            model = NOISE_MODELS[model_name]
            x_min = np.min(row_values)
            x_max = np.max(row_values)
            if model_name in ["log-normal", "exponential", "gamma"]:
                x_min = max(0.001, x_min)
            x = np.linspace(x_min, x_max, 200)
            y = model.pdf(x, *params)
            plt.plot(x, y, "k-", linewidth=2)

        plt.title(f"Row {row_idx + 1} | {model_name}")
        plt.xlabel("Amplitude")
        plt.ylabel("Density")
        plt.tight_layout()
        if filename_prefix:
            filename = f"{filename_prefix}_row_{row_idx + 1}.png"
        else:
            filename = f"row_{row_idx + 1}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()


def noise_fitting_step(input_csv_path: str, output_csv_path: str, plot_dir: str, filename_prefix: str = "") -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    df = dataframe_to_float(df)

    results, model_names = analyze_noise_models(df)
    selected = select_models_interactive(df, results, model_names)

    fit_df = fit_selected_models(df, results, selected, plot_dir)
    ensure_dir(os.path.dirname(output_csv_path))
    fit_df.to_csv(output_csv_path, index=False)

    plot_noise_fits(df, fit_df, plot_dir, filename_prefix=filename_prefix)
    return fit_df
