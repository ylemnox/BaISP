import os
import sys
from typing import List

import numpy as np

from utils import (
    ensure_dir,
    describe_rows,
    safe_int_list,
    infer_species_from_path,
    frequency_to_label,
    build_output_tag,
    sanitize_token,
)
from mat2csv_step import mat2csv_step
from preprocess_step import preprocess_step
from noise_fitting_step import noise_fitting_step


def confirm_step(message: str) -> bool:
    while True:
        ans = input(f"{message} [y/N]: ").strip().lower()
        if ans in ["y", "yes"]:
            return True
        if ans in ["", "n", "no"]:
            return False
        print("Please enter y or n")


def _normalize_path(raw: str) -> str:
    # Users sometimes paste shell-escaped paths; unescape common space escape.
    if "\\ " in raw:
        raw = raw.replace("\\ ", " ")
    return os.path.expanduser(raw)


def prompt_path(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    val = val if val else default
    return _normalize_path(val)


def prompt_float(prompt: str, default: float) -> float:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return float(default)
        try:
            return float(val)
        except ValueError:
            print("Please enter a valid number")


def enforce_tag_in_file(path: str, output_tag: str) -> str:
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    stem, ext = os.path.splitext(filename)
    if output_tag in stem:
        return path
    return os.path.join(directory, f"{stem}_{output_tag}{ext}")


def enforce_tag_in_dir(path: str, output_tag: str) -> str:
    base = os.path.basename(path.rstrip(os.sep))
    if output_tag in base:
        return path
    parent = os.path.dirname(path.rstrip(os.sep))
    return os.path.join(parent, f"{base}_{output_tag}")


def force_file_into_dir(path: str, target_dir: str) -> str:
    return os.path.join(target_dir, os.path.basename(path))


def force_dir_into_dir(path: str, target_dir: str) -> str:
    base = os.path.basename(path.rstrip(os.sep))
    return os.path.join(target_dir, base)


def confirm_metadata(mat_input_dir: str, frequency_hz: float) -> tuple[str, float]:
    inferred_species = infer_species_from_path(mat_input_dir)
    freq_label = frequency_to_label(frequency_hz)
    print(f"\nDetected species: {inferred_species}")
    print(f"Detected frequency: {freq_label}")
    if confirm_step("Is species/frequency correct?"):
        return inferred_species, frequency_hz

    species = input(f"Enter species [{inferred_species}]: ").strip()
    species = sanitize_token(species) if species else sanitize_token(inferred_species)
    frequency = prompt_float("Enter frequency (Hz)", frequency_hz)
    return species, frequency


def display_step1_summary(df, input_dir, output_path):
    rows = describe_rows(df, "Spike Time")
    print("\nSpike time list with non-NaN counts:")
    print("-" * 60)
    for idx, t, non_nan in rows:
        print(f"Row {idx + 1:>2} | Spike Time {t:.6f} | Non-NaN {non_nan}")
    print("\nInput dir:", input_dir)
    print("Output CSV:", output_path)


def main():
    print("BaISP (Interactive CLI)")
    print("BaISP: Bayesian Inference of Synaptic Plasticity")
    print("=" * 40)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_output_dir = os.path.join(script_dir, "output")
    mat_input_dir = prompt_path("Mat input directory", project_root)
    frequency_hz = prompt_float("Frequency (Hz)", 50.0)
    species, frequency_hz = confirm_metadata(mat_input_dir, frequency_hz)
    output_tag = build_output_tag(species, frequency_hz)
    run_output_dir = os.path.join(base_output_dir, output_tag)
    ensure_dir(run_output_dir)
    print(f"Using output tag: {output_tag}")
    print(f"Output subdirectory: {run_output_dir}")

    step1_default = os.path.join(run_output_dir, f"step1_{output_tag}_mat2csv.csv")
    step1_output = prompt_path("Step1 output CSV", step1_default)
    step1_output = force_file_into_dir(step1_output, run_output_dir)
    step1_output = enforce_tag_in_file(step1_output, output_tag)
    ensure_dir(os.path.dirname(step1_output))

    print("\nSTEP 1: Mat2Csv")
    df_step1 = mat2csv_step(mat_input_dir, step1_output, frequency_hz)
    display_step1_summary(df_step1, mat_input_dir, step1_output)

    if not confirm_step("Continue to STEP 2?"):
        print("Stopped after STEP 1")
        return

    print("\nSTEP 2: Preprocessing")
    print("Enter rows to remove by row number (1-based, comma-separated). Press Enter to keep all.")
    row_input = input("Rows to remove: ").strip()
    rows_to_remove: List[int] = []
    if row_input:
        rows = safe_int_list(row_input)
        rows_to_remove = [r - 1 for r in rows if r > 0]

    step2_default = os.path.join(run_output_dir, f"step2_{output_tag}_preprocessed.csv")
    step2_output = prompt_path("Step2 output CSV", step2_default)
    step2_output = force_file_into_dir(step2_output, run_output_dir)
    step2_output = enforce_tag_in_file(step2_output, output_tag)
    df_step2 = preprocess_step(step1_output, step2_output, rows_to_remove)

    print(f"Preprocessed CSV saved: {step2_output}")
    print(f"Rows remaining: {len(df_step2)}")

    if not confirm_step("Continue to STEP 3?"):
        print("Stopped after STEP 2")
        return

    print("\nSTEP 3: Noise Fitting")
    step3_default = os.path.join(run_output_dir, f"step3_{output_tag}_noise_fit.csv")
    plot_default = os.path.join(run_output_dir, f"noise_plots_{output_tag}")
    step3_output = prompt_path("Step3 output CSV", step3_default)
    plot_dir = prompt_path("Noise plot output dir", plot_default)
    step3_output = force_file_into_dir(step3_output, run_output_dir)
    plot_dir = force_dir_into_dir(plot_dir, run_output_dir)
    step3_output = enforce_tag_in_file(step3_output, output_tag)
    plot_dir = enforce_tag_in_dir(plot_dir, output_tag)
    fit_df = noise_fitting_step(step2_output, step3_output, plot_dir, filename_prefix=f"noise_{output_tag}")
    print(f"Noise fitting results saved: {step3_output}")
    print(f"Noise plots saved in: {plot_dir}")

    if confirm_step("Override noise parameters?"):
        for idx, row in fit_df.iterrows():
            current = row["param"]
            prompt = f"Row {int(row['row_index']) + 1} ({row['model']}) param [{current}]: "
            val = input(prompt).strip()
            if val:
                try:
                    fit_df.at[idx, "param"] = float(val)
                except ValueError:
                    print("Invalid number, keeping existing value")
        fit_df.to_csv(step3_output, index=False)
        print("Updated noise parameters saved")

    if not confirm_step("Continue to STEP 4?"):
        print("Stopped after STEP 3")
        return

    print("\nSTEP 4: Bayesian Formulation, MCMC Sampling")
    step4_default = os.path.join(run_output_dir, f"step4_{output_tag}_mcmc_results.csv")
    plot4_default = os.path.join(run_output_dir, f"posterior_distributions_{output_tag}.png")
    step4_output = prompt_path("Step4 output CSV", step4_default)
    plot_output = prompt_path("Posterior plot output file", plot4_default)
    step4_output = force_file_into_dir(step4_output, run_output_dir)
    plot_output = force_file_into_dir(plot_output, run_output_dir)
    step4_output = enforce_tag_in_file(step4_output, output_tag)
    plot_output = enforce_tag_in_file(plot_output, output_tag)

    if np.isnan(fit_df["param"]).any():
        print("Error: noise parameters contain NaN. Fix in step3 output before running step4.")
        return

    from mcmc_step import mcmc_step
    mcmc_step(step2_output, step3_output, step4_output, plot_output, frequency_hz)
    print(f"MCMC results saved: {step4_output}")
    print(f"Posterior plot saved: {plot_output}")


if __name__ == "__main__":
    main()
