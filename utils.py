import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.io
import h5py


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_mat_file(mat_file_path: str, variable_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load .mat file with scipy.io, fallback to h5py for v7.3."""
    try:
        if variable_names:
            return scipy.io.loadmat(mat_file_path, variable_names=variable_names)
        return scipy.io.loadmat(mat_file_path)
    except Exception as e:
        if "HDF reader" in str(e) or "matlab v7.3" in str(e).lower():
            mat_data: Dict[str, Any] = {}
            with h5py.File(mat_file_path, "r") as f:
                def extract_data(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if variable_names and name not in variable_names:
                            return
                        data = obj[()]
                        if data.dtype.kind == "O":
                            try:
                                if data.size == 1:
                                    ref = data.item()
                                    if isinstance(ref, h5py.Reference):
                                        data = f[ref][()]
                            except Exception:
                                pass
                        mat_data[name] = data
                if variable_names:
                    for key in variable_names:
                        if key in f:
                            extract_data(key, f[key])
                else:
                    f.visititems(extract_data)
            return mat_data
        raise


def list_mat_files(input_dir: str) -> List[str]:
    files = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".mat"):
            files.append(os.path.join(input_dir, name))
    return sorted(files)


def safe_float_list(text: str) -> List[float]:
    if not text.strip():
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    return out


def safe_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def dataframe_to_float(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def describe_rows(df: pd.DataFrame, time_col: str) -> List[Tuple[int, float, int]]:
    rows = []
    value_cols = [c for c in df.columns if c != time_col]
    for i, row in df.iterrows():
        values = row[value_cols].values
        non_nan = int(np.isfinite(values).sum())
        rows.append((i, float(row[time_col]), non_nan))
    return rows


def infer_species_from_path(path: str) -> str:
    """Infer species token from input path."""
    norm = path.replace("\\", "/").lower()
    candidates = ["human", "mouse", "rodent", "nhp", "macaque", "rat"]
    for c in candidates:
        if f"/{c}/" in norm or norm.endswith(f"/{c}") or f"_{c}_" in norm:
            return c
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    for p in parts:
        low = p.lower()
        if low in candidates:
            return low
    return "unknown"


def frequency_to_label(frequency_hz: float) -> str:
    """Convert frequency to compact filename-safe label, e.g. 10Hz or 12p5Hz."""
    if abs(frequency_hz - round(frequency_hz)) < 1e-9:
        return f"{int(round(frequency_hz))}Hz"
    text = f"{frequency_hz:.6f}".rstrip("0").rstrip(".")
    text = text.replace(".", "p")
    return f"{text}Hz"


def sanitize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9_-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def build_output_tag(species: str, frequency_hz: float) -> str:
    return f"{sanitize_token(species)}_{frequency_to_label(frequency_hz)}"
