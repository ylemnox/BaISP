# BaISP
<img src="/logo.png.png" width="170" align="right">

**BaISP** (Bayesian Inference of Synaptic Plasticity) is an interactive pipeline for short-term synaptic plasticity (STP) analysis from `.mat` files.

It automates the manual workflow described in `AGENTS.md` and follows the Bayesian inference framework described in:

- *Probabilistic inference of short-term synaptic plasticity in neocortical microcircuits* (Costa et al., 2013)
## What BaISP Does

BaISP runs 4 steps end-to-end:

1. `STEP 1` Mat2Csv: convert experiment `.mat` files into a single aligned CSV (`Spike Time` + experiment columns)
2. `STEP 2` Preprocessing: remove user-selected rows and fill `NaN` with row mean
3. `STEP 3` Noise fitting: fit row-wise noise models (`gaussian`, `log-normal`, `uniform`, `exponential`, `gamma`)
4. `STEP 4` Bayesian inference: run MCMC and export MAP parameters (`U`, `D`, `F`, `f`, `MAP`) per experiment

## Input Data Format

Each `.mat` file should contain:

- `amp_avg`: induction EPSP summary (`8 x 2` expected)
- `recovery_amp`: recovery EPSP rows (`N x 5` expected)

`recovery_amp` semantics used by BaISP:

- col 1: EPSP index (e.g., `9,10,11,12,...`)
- col 2: ignored
- col 3: ignored
- col 4: interval in **ms** after the 8th pulse
- col 5: EPSP value

## Spike Time Generation

BaISP generates times in **seconds**.

- Induction (first 8 pulses): `t_i = i / frequency_hz`, `i=0..7`
- Let `t8` be the 8th pulse time (`7 / frequency_hz`)
- For each recovery row:
  - `interval_s = recovery_amp[row, 3] / 1000`
  - `epsp_idx = recovery_amp[row, 0]`
  - `pulse_offset = epsp_idx - 9`
  - `t_recovery = t8 + interval_s + pulse_offset * (1 / frequency_hz)`

Then times and EPSP values are sorted together by spike time.

Notes:

- For `50 Hz`, all recovery rows are used.
- For non-`50 Hz`, BaISP currently keeps `250 ms` recovery rows (fallback: first 4 recovery rows if 250 ms rows are absent).

## Installation

Run from `pipeline/`.

```bash
cd pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run (CLI)

```bash
cd pipeline
source .venv/bin/activate
python run_pipeline.py
```

Or use the launcher:

```bash
cd pipeline
./run_baisp_cli.sh
```

CLI behavior:

- prompts for mat input folder and frequency
- infers species from path and asks for confirmation
- creates output subfolder: `output/{species}_{frequencyLabel}`
- runs step-by-step with user confirmation between steps

## Run (GUI)

Use Streamlit (do **not** run `python app.py` directly).

```bash
cd pipeline
source .venv/bin/activate
streamlit run app.py
```

Or use the launcher:

```bash
cd pipeline
./run_baisp_gui.sh
```

GUI highlights:

- Finder/Explorer folder browsing for input/output folders
- metadata confirmation gate (species + frequency)
- STEP 2 row-selection toggle buttons (4 columns)
- STEP 4 progress bar (`done/total` experiments)

## Output Files

All outputs are written under:

- `pipeline/output/{species}_{frequencyLabel}/`

Generated files:

- `step1_{tag}_mat2csv.csv`
- `step2_{tag}_preprocessed.csv`
- `step3_{tag}_noise_fit.csv`
- `noise_plots_{tag}/...png`
- `step4_{tag}_mcmc_results.csv`
- `posterior_distributions_{tag}.png`

Example tag:

- `human_10Hz`
- `mouse_50Hz`

## MCMC Defaults

Defined in `mcmc_step.py`:

- chains: `5`
- samples per chain: `10000`
- burn-in (`tune`): `2500`
- sampling logs are silenced to keep terminal output concise

## Common Issues

### `python app.py` shows many Streamlit warnings

Expected if run as plain Python. Use:

```bash
streamlit run app.py
```

### Pasted path with `\ ` fails in CLI

BaISP normalizes shell-escaped spaces in CLI input, but easiest is to paste normal path text (without backslash escapes).

### Step 1 is slow or killed

Typical causes:

- wrong folder path
- very large number of files
- low-memory environment

Try with a small subset first and verify one run end-to-end.

## Repository Structure

- `run_pipeline.py`: interactive CLI orchestrator
- `app.py`: Streamlit GUI orchestrator
- `mat2csv_step.py`: step 1
- `preprocess_step.py`: step 2
- `noise_fitting_step.py`: step 3
- `mcmc_step.py`: step 4
- `utils.py`: shared utilities
