import os
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from utils import ensure_dir, load_mat_file, list_mat_files


def _get_matrix(mat_data: Dict[str, Any], name: str) -> np.ndarray:
    if name not in mat_data:
        raise KeyError(f"Missing matrix '{name}'")
    data = mat_data[name]
    if hasattr(data, "T"):
        return np.array(data)
    return np.array(data)


def _compute_spike_times(frequency_hz: float, recovery_amp: np.ndarray, n_recovery: int = None) -> np.ndarray:
    # 8 induction spikes
    induction_interval = 1.0 / frequency_hz
    induction_times = np.array([i * induction_interval for i in range(8)], dtype=float)
    t8 = induction_times[-1]

    # recovery_amp semantics:
    # col 1: EPSP index (9..12), col 4: interval in ms after 8th pulse, col 5: EPSP value
    # spike time per row = t8 + interval_s + (epsp_index - 9) * protocol_dt
    interval_s = recovery_amp[:, 3].astype(float) / 1000.0
    epsp_index = recovery_amp[:, 0].astype(float)
    pulse_offset_index = np.round(epsp_index - 9).astype(int)
    recovery_times = t8 + interval_s + pulse_offset_index * induction_interval

    spike_times = np.concatenate([induction_times, recovery_times])
    return spike_times


def _build_experiment_series(mat_data: Dict[str, Any], frequency_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    amp_avg = _get_matrix(mat_data, "amp_avg")
    recovery_amp = _get_matrix(mat_data, "recovery_amp")

    # Normalize shapes if transposed in file
    if amp_avg.ndim == 2 and amp_avg.shape[0] != 8 and amp_avg.shape[1] == 8:
        amp_avg = amp_avg.T
    if recovery_amp.ndim == 2 and recovery_amp.shape[1] != 5 and recovery_amp.shape[0] == 5:
        recovery_amp = recovery_amp.T

    if amp_avg.ndim != 2 or amp_avg.shape[0] < 8:
        raise ValueError("amp_avg must be a 2D array with at least 8 rows")
    if recovery_amp.ndim != 2 or recovery_amp.shape[1] < 5:
        raise ValueError("recovery_amp must be a 2D array with at least 5 columns")

    # EPSP values: amp_avg col 2 for first 8 spikes, recovery_amp col 5 for rest
    # Use EPSP index column (col 1) to align induction values
    amp_avg_sorted = amp_avg[np.argsort(amp_avg[:, 0])]
    epsp_induction = amp_avg_sorted[:8, 1].astype(float)

    # Recovery EPSP values follow recovery_amp rows; spike times are derived from recovery_amp col 4
    if frequency_hz == 50:
        epsp_recovery = recovery_amp[:, 4].astype(float)
        spike_times = _compute_spike_times(frequency_hz, recovery_amp)
    else:
        # Select recovery rows at 250 ms for non-50Hz
        rec_times = recovery_amp[:, 3].astype(float)
        mask = np.isclose(rec_times, 250.0)
        recovery_rows = recovery_amp[mask]
        if recovery_rows.size == 0:
            # fallback to first 4 rows
            recovery_rows = recovery_amp[: min(len(recovery_amp), 4)]
        # sort by EPSP index column
        recovery_rows = recovery_rows[np.argsort(recovery_rows[:, 0])]
        epsp_recovery = recovery_rows[:, 4].astype(float)
        spike_times = _compute_spike_times(frequency_hz, recovery_rows, n_recovery=len(epsp_recovery))
    epsp_values = np.concatenate([epsp_induction, epsp_recovery])

    if len(spike_times) != len(epsp_values):
        raise ValueError("Spike time count does not match EPSP value count")

    # sort by spike time
    order = np.argsort(spike_times)
    spike_times_sorted = spike_times[order]
    epsp_sorted = epsp_values[order]

    # If duplicate spike times exist (unexpected), aggregate by mean to ensure unique index
    if len(np.unique(spike_times_sorted)) != len(spike_times_sorted):
        uniq_times, inverse = np.unique(spike_times_sorted, return_inverse=True)
        sums = np.zeros(len(uniq_times), dtype=float)
        counts = np.zeros(len(uniq_times), dtype=int)
        for i, uidx in enumerate(inverse):
            val = epsp_sorted[i]
            if np.isfinite(val):
                sums[uidx] += val
                counts[uidx] += 1
        means = np.array([sums[i] / counts[i] if counts[i] > 0 else np.nan for i in range(len(uniq_times))])
        spike_times_sorted = uniq_times
        epsp_sorted = means

    return spike_times_sorted, epsp_sorted


def mat2csv_step(mat_input_dir: str, output_csv_path: str, frequency_hz: float, verbose: bool = True) -> pd.DataFrame:
    ensure_dir(os.path.dirname(output_csv_path))

    mat_files = list_mat_files(mat_input_dir)
    limit_env = os.environ.get("PIPELINE_LIMIT")
    if limit_env:
        try:
            limit = int(limit_env)
            mat_files = mat_files[:limit]
        except ValueError:
            pass
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_input_dir}")

    series_list = []

    if verbose:
        print(f"Found {len(mat_files)} .mat files")
    for idx, mat_path in enumerate(mat_files, start=1):
        if verbose:
            print(f"[{idx}/{len(mat_files)}] Loading {os.path.basename(mat_path)}")
        mat_data = load_mat_file(mat_path, variable_names=["amp_avg", "recovery_amp"])
        spike_times, epsp_values = _build_experiment_series(mat_data, frequency_hz)

        col_name = Path(mat_path).stem
        series = pd.Series(epsp_values, index=spike_times, name=col_name)
        series_list.append(series)

        del mat_data, series
        gc.collect()

    if not series_list:
        raise RuntimeError("No data loaded from mat files")

    combined_df = pd.concat(series_list, axis=1, join="outer")
    combined_df.index.name = "Spike Time"
    combined_df = combined_df.sort_index().reset_index()
    combined_df.to_csv(output_csv_path, index=False)
    return combined_df
