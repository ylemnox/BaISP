import os
from typing import List

import numpy as np
import pandas as pd

from utils import ensure_dir, dataframe_to_float


def preprocess_step(input_csv_path: str, output_csv_path: str, rows_to_remove: List[int]) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    df = dataframe_to_float(df)

    if "Spike Time" not in df.columns:
        raise ValueError("Expected 'Spike Time' column in input CSV")

    # Remove rows by index (0-based)
    if rows_to_remove:
        df = df.drop(index=rows_to_remove)

    # Fill NaNs with row mean (excluding Spike Time)
    value_cols = [c for c in df.columns if c != "Spike Time"]

    def fill_row(row: pd.Series) -> pd.Series:
        values = row[value_cols].values.astype(float)
        if np.isfinite(values).sum() == 0:
            return row
        mean_val = np.nanmean(values)
        for c in value_cols:
            if pd.isna(row[c]):
                row[c] = mean_val
        return row

    df = df.apply(fill_row, axis=1)

    # Drop any rows that are still all-NaN in value columns
    mask_all_nan = df[value_cols].isna().all(axis=1)
    df = df[~mask_all_nan].reset_index(drop=True)

    ensure_dir(os.path.dirname(output_csv_path))
    df.to_csv(output_csv_path, index=False)
    return df
