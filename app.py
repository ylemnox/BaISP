import os
import sys
import subprocess
import base64
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from mat2csv_step import mat2csv_step
from preprocess_step import preprocess_step
from noise_fitting_step import (
    NOISE_MODELS,
    analyze_noise_models,
    fit_selected_models,
    plot_noise_fits,
)
from utils import (
    describe_rows,
    ensure_dir,
    infer_species_from_path,
    frequency_to_label,
    build_output_tag,
    sanitize_token,
)


def is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def apply_ui_style_overrides() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-size: 18px;
        }
        [data-testid="stCaptionContainer"] p {
            font-size: 16px !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            font-size: 18px !important;
        }
        .stAlert p {
            font-size: 18px !important;
            line-height: 1.4;
        }
        button[kind],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"] {
            font-size: 16px !important;
        }
        [data-testid="stSidebar"] * {
            font-size: 1.1rem !important;
            line-height: 1.3 !important;
        }
        .baisp-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 0.25rem;
        }
        .baisp-logo {
            height: 88px;
            width: auto;
            display: block;
            border-radius: 8px;
        }
        .baisp-title {
            font-size: 2.4rem;
            font-weight: 700;
            line-height: 1;
        }
        .baisp-subtitle {
            font-size: 1rem;
            color: #666;
            margin-top: 4px;
            line-height: 1.2;
        }
        .baisp-author {
            font-size: 0.95rem;
            color: #777;
            margin-top: 2px;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        st.markdown(
            f"""
            <div class="baisp-header">
                <img class="baisp-logo" src="data:image/png;base64,{encoded}" />
                <div class="baisp-titles">
                    <div class="baisp-title">BaISP</div>
                    <div class="baisp-subtitle">Bayesian Inference of Synaptic Plasticity</div>
                    <div class="baisp-author">Created by ylemnox | Assisted by CodeX</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("BaISP")
        st.caption("Bayesian Inference of Synaptic Plasticity")


def normalize_path(raw: str) -> str:
    if "\\ " in raw:
        raw = raw.replace("\\ ", " ")
    return os.path.expanduser(raw.strip())


def default_paths() -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    output_dir = os.path.join(script_dir, "output")
    return {
        "project_root": project_root,
        "output_dir": output_dir,
    }


def pick_directory_dialog(initial_dir: str) -> str | None:
    """Open native folder picker (Finder/Explorer) and return selected path."""
    # macOS Finder dialog (most reliable when Streamlit runs outside a GUI mainloop)
    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "Select folder")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None
            picked = result.stdout.strip()
            return picked if picked else None
        except Exception:
            return None

    # Windows folder picker
    if os.name == "nt":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$f.SelectedPath = '" + (initial_dir or "").replace("'", "''") + "';"
            "$ok = $f.ShowDialog();"
            "if($ok -eq [System.Windows.Forms.DialogResult]::OK){$f.SelectedPath}"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                check=False,
            )
            picked = result.stdout.strip()
            return picked if picked else None
        except Exception:
            return None

    # Linux fallback: zenity
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        picked = result.stdout.strip()
        return picked if picked else None
    except Exception:
        return None


def derived_output_paths(output_dir: str, species: str, frequency_hz: float) -> Dict[str, str]:
    output_tag = build_output_tag(species, frequency_hz)
    run_output_dir = os.path.join(output_dir, output_tag)
    return {
        "output_tag": output_tag,
        "run_output_dir": run_output_dir,
        "step1_output": os.path.join(run_output_dir, f"step1_{output_tag}_mat2csv.csv"),
        "step2_output": os.path.join(run_output_dir, f"step2_{output_tag}_preprocessed.csv"),
        "step3_output": os.path.join(run_output_dir, f"step3_{output_tag}_noise_fit.csv"),
        "noise_plot_dir": os.path.join(run_output_dir, f"noise_plots_{output_tag}"),
        "step4_output": os.path.join(run_output_dir, f"step4_{output_tag}_mcmc_results.csv"),
        "step4_plot": os.path.join(run_output_dir, f"posterior_distributions_{output_tag}.png"),
    }


def init_state() -> None:
    keys = {
        "step1_df": None,
        "step2_df": None,
        "step2_remove_rows": [],
        "step2_rows_signature": "",
        "step3_fit_df": None,
        "step3_results": None,
        "step3_model_names": None,
        "step3_top3_df": None,
        "step4_df": None,
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_csv_if_exists(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def top3_table(df_step2: pd.DataFrame, results: Dict[int, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    spike_times = df_step2["Spike Time"].values
    for row_idx, spike_t in enumerate(spike_times):
        ranked: List[Tuple[str, float]] = []
        for model_name, item in results[row_idx].items():
            if item.get("ok"):
                ranked.append((model_name, float(item["p_value"])))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:3]
        rows.append(
            {
                "row_index": row_idx,
                "spike_time": float(spike_t),
                "top1_model": top[0][0] if len(top) > 0 else "",
                "top1_p": top[0][1] if len(top) > 0 else np.nan,
                "top2_model": top[1][0] if len(top) > 1 else "",
                "top2_p": top[1][1] if len(top) > 1 else np.nan,
                "top3_model": top[2][0] if len(top) > 2 else "",
                "top3_p": top[2][1] if len(top) > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def render_step1(mat_input_dir: str, frequency_hz: float, step1_output: str, disabled: bool = False) -> None:
    st.subheader("STEP 1: Mat2Csv")

    if st.button("Run STEP 1", type="primary", disabled=disabled):
        try:
            ensure_dir(os.path.dirname(step1_output))
            with st.spinner("Converting mat files..."):
                df_step1 = mat2csv_step(mat_input_dir, step1_output, frequency_hz, verbose=False)
            st.session_state["step1_df"] = df_step1
            st.success("STEP 1 completed")
        except Exception as e:
            st.error(f"STEP 1 failed: {e}")

    df = st.session_state["step1_df"]
    if df is None:
        df = load_csv_if_exists(step1_output)
        if df is not None:
            st.session_state["step1_df"] = df

    if df is not None:
        rows = describe_rows(df, "Spike Time")
        summary_df = pd.DataFrame(
            [{"row": i + 1, "spike_time": t, "non_nan": non_nan} for i, t, non_nan in rows]
        )
        st.caption(f"Input dir: `{mat_input_dir}`")
        st.caption(f"Output CSV: `{step1_output}`")
        st.dataframe(summary_df, width="stretch", hide_index=True)


def render_step2(step1_output: str, step2_output: str, disabled: bool = False) -> None:
    st.subheader("STEP 2: Preprocessing")

    step1_df = st.session_state["step1_df"]
    if step1_df is None:
        step1_df = load_csv_if_exists(step1_output)
        if step1_df is not None:
            st.session_state["step1_df"] = step1_df

    if step1_df is None:
        st.info("Run STEP 1 first or provide an existing STEP 1 CSV.")
        return

    rows = describe_rows(step1_df, "Spike Time")
    signature = f"{len(rows)}|{step1_output}"
    if st.session_state["step2_rows_signature"] != signature:
        st.session_state["step2_rows_signature"] = signature
        st.session_state["step2_remove_rows"] = []

    st.caption("Click row buttons to toggle removal.")
    selected_rows = set(st.session_state["step2_remove_rows"])

    # Stable 4-column button grid
    for start in range(0, len(rows), 4):
        cols = st.columns(4, gap="small")
        chunk = rows[start:start + 4]
        for j, (idx, spike_t, non_nan) in enumerate(chunk):
            selected = idx in selected_rows
            label = f"Row {idx + 1} | t={spike_t:.6f} | n={non_nan}"
            if selected:
                label = f"[REMOVE] {label}"
            if cols[j].button(
                label,
                key=f"row_remove_btn_{idx}",
                type="primary" if selected else "secondary",
                disabled=disabled,
                width="stretch",
            ):
                if selected:
                    selected_rows.remove(idx)
                else:
                    selected_rows.add(idx)
                st.session_state["step2_remove_rows"] = sorted(selected_rows)
                st.rerun()

    c1, c2 = st.columns(2)
    c1.write(f"Selected rows to remove: {len(selected_rows)}")
    if c2.button("Clear Row Selection", disabled=disabled, width="stretch"):
        st.session_state["step2_remove_rows"] = []
        st.rerun()

    if st.button("Run STEP 2", disabled=disabled, type="primary"):
        try:
            rows_to_remove: List[int] = sorted(st.session_state["step2_remove_rows"])

            with st.spinner("Preprocessing rows..."):
                df_step2 = preprocess_step(step1_output, step2_output, rows_to_remove)
            st.session_state["step2_df"] = df_step2
            st.success("STEP 2 completed")
        except Exception as e:
            st.error(f"STEP 2 failed: {e}")

    df = st.session_state["step2_df"]
    if df is None:
        df = load_csv_if_exists(step2_output)
        if df is not None:
            st.session_state["step2_df"] = df

    if df is not None:
        st.caption(f"Output CSV: `{step2_output}`")
        st.write(f"Rows remaining: {len(df)}")
        st.dataframe(df, width="stretch", height=280)


def render_step3(step2_output: str, step3_output: str, noise_plot_dir: str, output_tag: str, disabled: bool = False) -> None:
    st.subheader("STEP 3: Noise Fitting")

    df_step2 = st.session_state["step2_df"]
    if df_step2 is None:
        df_step2 = load_csv_if_exists(step2_output)
        if df_step2 is not None:
            st.session_state["step2_df"] = df_step2

    if df_step2 is None:
        st.info("Run STEP 2 first or provide an existing STEP 2 CSV.")
        return

    if st.button("Analyze Noise Models", disabled=disabled):
        try:
            with st.spinner("Analyzing candidate models..."):
                results, model_names = analyze_noise_models(df_step2)
            st.session_state["step3_results"] = results
            st.session_state["step3_model_names"] = model_names
            st.session_state["step3_top3_df"] = top3_table(df_step2, results)
            st.success("Model analysis complete")
        except Exception as e:
            st.error(f"Noise model analysis failed: {e}")

    top3_df = st.session_state["step3_top3_df"]
    results = st.session_state["step3_results"]
    model_names = st.session_state["step3_model_names"]

    if top3_df is not None:
        st.dataframe(top3_df, width="stretch", hide_index=True, height=280)

    if results is None or model_names is None:
        return

    mode = st.radio("Model selection mode", ["Same model for all rows", "Select per row"], index=0)

    selected_models: Dict[int, str] = {}
    if mode == "Same model for all rows":
        avg_scores = {}
        for name in model_names:
            pvals = [results[i][name]["p_value"] for i in results if results[i][name].get("ok")]
            avg_scores[name] = np.mean(pvals) if pvals else -1.0
        default_model = max(avg_scores.items(), key=lambda x: x[1])[0]
        chosen = st.selectbox("Model", options=model_names, index=model_names.index(default_model))
        for row_idx in results:
            selected_models[row_idx] = chosen
    else:
        selection_df = top3_df[["row_index", "spike_time", "top1_model"]].copy()
        selection_df["selected_model"] = selection_df["top1_model"]
        edited = st.data_editor(selection_df, width="stretch", hide_index=True)
        for _, row in edited.iterrows():
            row_idx = int(row["row_index"])
            model = str(row["selected_model"]).strip().lower()
            if model in model_names:
                selected_models[row_idx] = model
            else:
                selected_models[row_idx] = str(row["top1_model"]).strip().lower()

    if st.button("Run STEP 3", disabled=disabled):
        try:
            with st.spinner("Fitting selected models and generating plots..."):
                fit_df = fit_selected_models(df_step2, results, selected_models, noise_plot_dir)
                ensure_dir(os.path.dirname(step3_output))
                fit_df.to_csv(step3_output, index=False)
                plot_noise_fits(df_step2, fit_df, noise_plot_dir, filename_prefix=f"noise_{output_tag}")
            st.session_state["step3_fit_df"] = fit_df
            st.success("STEP 3 completed")
        except Exception as e:
            st.error(f"STEP 3 failed: {e}")

    fit_df = st.session_state["step3_fit_df"]
    if fit_df is None:
        fit_df = load_csv_if_exists(step3_output)
        if fit_df is not None:
            st.session_state["step3_fit_df"] = fit_df

    if fit_df is not None:
        st.caption(f"Output CSV: `{step3_output}`")
        st.caption(f"Plots: `{noise_plot_dir}`")
        edit_cols = ["row_index", "spike_time", "model", "p_value", "param"]
        editable = fit_df[edit_cols].copy()
        edited_params = st.data_editor(editable, width="stretch", hide_index=True)
        if st.button("Save Parameter Overrides", disabled=disabled):
            try:
                merged = fit_df.copy()
                merged["param"] = pd.to_numeric(edited_params["param"], errors="coerce")
                merged.to_csv(step3_output, index=False)
                st.session_state["step3_fit_df"] = merged
                st.success("Saved parameter overrides")
            except Exception as e:
                st.error(f"Failed to save overrides: {e}")


def render_step4(step2_output: str, step3_output: str, step4_output: str, step4_plot: str, frequency_hz: float, disabled: bool = False) -> None:
    st.subheader("STEP 4: Bayesian Formulation, MCMC")

    if st.button("Run STEP 4", disabled=disabled):
        try:
            fit_df = load_csv_if_exists(step3_output)
            if fit_df is None:
                raise ValueError("STEP 3 output CSV not found")
            if fit_df["param"].isna().any():
                raise ValueError("Noise parameters contain NaN. Fix STEP 3 parameters first.")

            from mcmc_step import mcmc_step

            progress_text = st.empty()
            progress_bar = st.progress(0, text="Sampling 0/0")

            def on_progress(done: int, total: int, name: str) -> None:
                total_safe = total if total > 0 else 1
                ratio = float(done) / float(total_safe)
                msg = f"Sampling {done}/{total}"
                if name:
                    msg += f" ({name})"
                progress_text.write(msg)
                progress_bar.progress(ratio, text=msg)

            with st.spinner("Running MCMC (this may take time)..."):
                step4_df = mcmc_step(
                    step2_output,
                    step3_output,
                    step4_output,
                    step4_plot,
                    frequency_hz,
                    progress_callback=on_progress,
                )
            progress_bar.progress(1.0, text="Sampling complete")
            st.session_state["step4_df"] = step4_df
            st.success("STEP 4 completed")
        except Exception as e:
            st.error(f"STEP 4 failed: {e}")

    df = st.session_state["step4_df"]
    if df is None:
        df = load_csv_if_exists(step4_output)
        if df is not None:
            st.session_state["step4_df"] = df

    if df is not None:
        st.caption(f"Output CSV: `{step4_output}`")
        st.dataframe(df, width="stretch", hide_index=True)

    if os.path.exists(step4_plot):
        st.image(step4_plot, caption="Posterior Distributions", width="stretch")


def main() -> None:
    st.set_page_config(page_title="BaISP", layout="wide")
    apply_ui_style_overrides()
    init_state()
    defaults = default_paths()

    render_header()

    with st.sidebar:
        st.header("Inputs")
        if "mat_input_dir" not in st.session_state:
            st.session_state["mat_input_dir"] = defaults["project_root"]
        if "output_base_dir" not in st.session_state:
            st.session_state["output_base_dir"] = defaults["output_dir"]
        if "species_name" not in st.session_state:
            st.session_state["species_name"] = "unknown"
        if "species_source_dir" not in st.session_state:
            st.session_state["species_source_dir"] = ""

        st.subheader("Mat Input Folder")
        st.caption(f"`{st.session_state['mat_input_dir']}`")
        if st.button("Browse Mat Folder", key="browse_mat_input", width="stretch"):
            picked = pick_directory_dialog(st.session_state["mat_input_dir"])
            if picked:
                st.session_state["mat_input_dir"] = normalize_path(picked)
                st.rerun()
            else:
                st.warning("No folder selected or folder dialog is not available on this environment.")

        frequency_hz = float(st.number_input("Frequency (Hz)", min_value=0.1, value=50.0, step=0.1))

        inferred_species = infer_species_from_path(st.session_state["mat_input_dir"])
        if st.session_state["species_source_dir"] != st.session_state["mat_input_dir"]:
            st.session_state["species_name"] = inferred_species
            st.session_state["species_source_dir"] = st.session_state["mat_input_dir"]

        st.subheader("Metadata")
        st.caption(f"Inferred species: `{inferred_species}`")
        st.session_state["species_name"] = sanitize_token(
            st.text_input("Species (editable)", value=st.session_state["species_name"])
        )
        st.caption(f"Frequency label: `{frequency_to_label(frequency_hz)}`")
        meta_key = f"{st.session_state['mat_input_dir']}|{st.session_state['species_name']}|{frequency_hz}"
        if st.session_state.get("meta_key_confirmed", "") != meta_key:
            st.session_state["confirm_metadata"] = False
            st.session_state["meta_key_confirmed"] = meta_key
        metadata_confirmed = st.checkbox(
            "Confirm species and frequency are correct",
            key="confirm_metadata",
        )
        if not metadata_confirmed:
            st.warning("Confirm metadata to enable STEP 1-4 execution.")

        st.subheader("Output Folder")
        st.caption(f"`{st.session_state['output_base_dir']}`")
        if st.button("Browse Output Folder", key="browse_output_base", width="stretch"):
            picked = pick_directory_dialog(st.session_state["output_base_dir"])
            if picked:
                st.session_state["output_base_dir"] = normalize_path(picked)
                st.rerun()
            else:
                st.warning("No folder selected or folder dialog is not available on this environment.")

        mat_input_dir = st.session_state["mat_input_dir"]
        output_paths = derived_output_paths(
            st.session_state["output_base_dir"],
            st.session_state["species_name"],
            frequency_hz,
        )
        output_tag = output_paths["output_tag"]
        run_output_dir = output_paths["run_output_dir"]
        step1_output = output_paths["step1_output"]
        step2_output = output_paths["step2_output"]
        step3_output = output_paths["step3_output"]
        noise_plot_dir = output_paths["noise_plot_dir"]
        step4_output = output_paths["step4_output"]
        step4_plot = output_paths["step4_plot"]

        st.caption(f"Run output folder: `{run_output_dir}`")
        st.caption(f"STEP 1 CSV: `{step1_output}`")
        st.caption(f"STEP 2 CSV: `{step2_output}`")
        st.caption(f"STEP 3 CSV: `{step3_output}`")
        st.caption(f"Noise plots: `{noise_plot_dir}`")
        st.caption(f"STEP 4 CSV: `{step4_output}`")
        st.caption(f"STEP 4 plot: `{step4_plot}`")

    render_step1(mat_input_dir, frequency_hz, step1_output, disabled=not metadata_confirmed)
    st.divider()
    render_step2(step1_output, step2_output, disabled=not metadata_confirmed)
    st.divider()
    render_step3(step2_output, step3_output, noise_plot_dir, output_tag, disabled=not metadata_confirmed)
    st.divider()
    render_step4(step2_output, step3_output, step4_output, step4_plot, frequency_hz, disabled=not metadata_confirmed)


if __name__ == "__main__":
    if is_streamlit_runtime():
        main()
    else:
        print("Run this GUI with Streamlit, not plain Python.")
        print("Command: streamlit run app.py")
        sys.exit(0)
