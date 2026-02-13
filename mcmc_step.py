import os
import io
import logging
import warnings
import contextlib
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func

try:
    import pymc as pm
    import arviz as az  # noqa: F401
    PYMC_AVAILABLE = True
except Exception:
    PYMC_AVAILABLE = False

from utils import ensure_dir, dataframe_to_float


@contextlib.contextmanager
def quiet_sampling_logs():
    """Temporarily silence noisy sampler/library logs."""
    targets = ["pymc", "arviz", "pytensor"]
    old_levels = {}
    for name in targets:
        logger = logging.getLogger(name)
        old_levels[name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
    finally:
        for name in targets:
            logging.getLogger(name).setLevel(old_levels[name])


def etm_model(theta: np.ndarray, spike_times: np.ndarray, frequency_hz: float) -> np.ndarray:
    D, F, U, f = theta

    R = np.zeros(len(spike_times))
    u = np.zeros(len(spike_times))
    PSP = np.zeros(len(spike_times))
    R[0] = 1.0
    u[0] = U
    PSP[0] = R[0] * u[0]

    for i in range(1, len(spike_times)):
        if frequency_hz == 50:
            first_recov = [8, 12, 16, 20, 24, 28, 32, 36, 40]
            if i in first_recov:
                dt = spike_times[i] - spike_times[7]
            else:
                dt = spike_times[i] - spike_times[i - 1]
        else:
            if i == 8:
                dt = spike_times[i] - spike_times[7]
            else:
                dt = spike_times[i] - spike_times[i - 1]

        R[i] = 1 - (1 - R[i - 1] * (1 - u[i - 1])) * np.exp(-dt / D)
        u[i] = U + (u[i - 1] + f * (1 - u[i - 1]) - U) * np.exp(-dt / F)
        PSP[i] = R[i] * u[i]

    return PSP


def log_likelihood(theta, data, model_psp, noise_models: List[str], noise_params: List[float]):
    log_lik = 0.0
    for i in range(len(data)):
        d_i = data[i]
        model_i = model_psp[i]
        model_name = noise_models[i]
        param_i = noise_params[i]

        if model_name == "gaussian":
            sigma_i = param_i
            log_lik += -0.5 * np.log(2 * np.pi * sigma_i ** 2) - (d_i - model_i) ** 2 / (2 * sigma_i ** 2)
        elif model_name == "gamma":
            k = param_i
            theta_i = model_i / k
            if d_i > 0 and theta_i > 0:
                log_lik += (k - 1) * np.log(d_i) - d_i / theta_i - k * np.log(theta_i) - np.log(gamma_func(k))
            else:
                return -np.inf
        elif model_name == "log-normal":
            sigma_i = param_i
            if d_i > 0 and model_i > 0:
                log_lik += -np.log(d_i * sigma_i * np.sqrt(2 * np.pi)) - (np.log(d_i) - np.log(model_i)) ** 2 / (2 * sigma_i ** 2)
            else:
                return -np.inf
        elif model_name == "exponential":
            scale = param_i
            if d_i >= 0 and scale > 0:
                log_lik += -np.log(scale) - d_i / scale
            else:
                return -np.inf
        elif model_name == "uniform":
            width = param_i
            if 0 <= d_i <= width:
                log_lik += -np.log(width)
            else:
                return -np.inf
        else:
            return -np.inf

    return log_lik


def log_prior(theta):
    D, F, U, f = theta
    if not (0 <= D <= 2 and 0 <= F <= 2 and 0 <= U <= 1 and 0 <= f <= 1):
        return -np.inf
    return -np.log(2) - np.log(2) - np.log(1) - np.log(1)


def log_posterior(theta, data, spike_times, frequency_hz, noise_models, noise_params):
    try:
        model_psp = etm_model(theta, spike_times, frequency_hz)
    except Exception:
        return -np.inf

    lp = log_prior(theta)
    if lp == -np.inf:
        return -np.inf

    ll = log_likelihood(theta, data, model_psp, noise_models, noise_params)
    return lp + ll


def custom_slice_sampler(log_pdf, initial_theta, widths, n_samples, burn_in=2500):
    def slice_sample_step(x, w, log_pdf_func):
        y = log_pdf_func(x) + np.log(np.random.uniform(0, 1))
        xl = x - w * np.random.uniform(0, 1)
        xr = xl + w
        while log_pdf_func(xl) > y:
            xl -= w
        while log_pdf_func(xr) > y:
            xr += w
        while True:
            x_new = xl + (xr - xl) * np.random.uniform(0, 1)
            if log_pdf_func(x_new) > y:
                return x_new
            if x_new < x:
                xl = x_new
            else:
                xr = x_new

    samples = []
    current_theta = initial_theta.copy()

    for i in range(n_samples + burn_in):
        for j in range(len(current_theta)):
            def single_param_log_pdf(param_val):
                theta_temp = current_theta.copy()
                theta_temp[j] = param_val
                return log_pdf(theta_temp)
            current_theta[j] = slice_sample_step(current_theta[j], widths[j], single_param_log_pdf)
        if i >= burn_in:
            samples.append(current_theta.copy())

    return np.array(samples)


def run_mcmc_chains(data, spike_times, frequency_hz, noise_models, noise_params, n_chains=5, n_samples=10000, burn_in=2500):
    use_pymc = PYMC_AVAILABLE and len(set(noise_models)) == 1 and noise_models[0] in ["gaussian", "gamma", "log-normal"]

    if use_pymc:
        model_name = noise_models[0]
        try:
            import pytensor.tensor as pt
            from pytensor.compile import wrap_py

            @wrap_py(itypes=[pt.dvector], otypes=[pt.dvector])
            def etm_model_op(theta_tensor):
                theta_np = np.array(theta_tensor)
                return etm_model(theta_np, spike_times, frequency_hz).astype("float64")

            with pm.Model() as model:
                D = pm.Uniform("D", lower=0, upper=2)
                F = pm.Uniform("F", lower=0, upper=2)
                U = pm.Uniform("U", lower=0, upper=1)
                f = pm.Uniform("f", lower=0, upper=1)
                theta = pt.stack([D, F, U, f])
                model_psp = etm_model_op(theta)

                if model_name == "gaussian":
                    for i in range(len(data)):
                        pm.Normal(f"obs_{i}", mu=model_psp[i], sigma=noise_params[i], observed=data[i])
                elif model_name == "gamma":
                    for i in range(len(data)):
                        k = noise_params[i]
                        pm.Gamma(f"obs_{i}", alpha=k, beta=model_psp[i] / k, observed=data[i])
                elif model_name == "log-normal":
                    for i in range(len(data)):
                        sigma_i = noise_params[i]
                        pm.LogNormal(f"obs_{i}", mu=pt.log(model_psp[i]), sigma=sigma_i, observed=data[i])

                with quiet_sampling_logs():
                    trace = pm.sample(
                        n_samples,
                        tune=burn_in,
                        chains=n_chains,
                        cores=1,
                        step=pm.Slice(),
                        return_inferencedata=True,
                        progressbar=False,
                        compute_convergence_checks=False,
                        random_seed=42,
                    )

            all_samples = []
            for chain in range(n_chains):
                chain_samples = np.column_stack([
                    trace.posterior["D"].values[chain],
                    trace.posterior["F"].values[chain],
                    trace.posterior["U"].values[chain],
                    trace.posterior["f"].values[chain],
                ])
                all_samples.append(chain_samples)
            return all_samples
        except Exception:
            pass

    widths = [1, 1, 2, 2]
    all_samples = []

    def log_pdf(theta):
        return log_posterior(theta, data, spike_times, frequency_hz, noise_models, noise_params)

    for chain in range(n_chains):
        initial_theta = np.array([
            np.random.uniform(0.1, 1.9),
            np.random.uniform(0.1, 1.9),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
        ])
        samples = custom_slice_sampler(log_pdf, initial_theta, widths, n_samples, burn_in)
        all_samples.append(samples)

    return all_samples


def find_map_estimate(all_samples, data, spike_times, frequency_hz, noise_models, noise_params):
    best_posterior = -np.inf
    best_theta = None

    def log_pdf(theta):
        return log_posterior(theta, data, spike_times, frequency_hz, noise_models, noise_params)

    for chain_samples in all_samples:
        for sample in chain_samples:
            posterior_val = log_pdf(sample)
            if posterior_val > best_posterior:
                best_posterior = posterior_val
                best_theta = sample.copy()

    return best_theta, best_posterior


def plot_posterior_distributions(all_samples_list, output_path: str, param_names=None):
    if param_names is None:
        param_names = ["D", "F", "U", "f"]

    param_ranges = [(0, 2), (0, 2), (0, 1), (0, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for param_idx, ax in enumerate(axes):
        for exp_samples in all_samples_list:
            combined_samples = np.vstack(exp_samples)
            counts, bins = np.histogram(combined_samples[:, param_idx], bins=50, range=param_ranges[param_idx], density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers, counts, color="lightblue", alpha=0.7, linewidth=1)

        if all_samples_list:
            all_param_samples = []
            for exp_samples in all_samples_list:
                combined_samples = np.vstack(exp_samples)
                all_param_samples.extend(combined_samples[:, param_idx])
            counts, bins = np.histogram(all_param_samples, bins=50, range=param_ranges[param_idx], density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers, counts, color="darkblue", linewidth=2, label="Average")

        ax.set_xlabel(f"Parameter {param_names[param_idx]}")
        ax.set_ylabel("Posterior")
        ax.set_title(f"Posterior for {param_names[param_idx]}")
        ax.grid(True, alpha=0.3)
        if param_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def load_experimental_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    df = dataframe_to_float(df)
    if "Spike Time" not in df.columns:
        raise ValueError("Expected 'Spike Time' column in input CSV")

    spike_times = df["Spike Time"].values.astype(float)
    value_cols = [c for c in df.columns if c != "Spike Time"]
    psp_data = df[value_cols].values.astype(float)
    return spike_times, psp_data, value_cols


def mcmc_step(
    input_csv_path: str,
    noise_fit_csv_path: str,
    output_csv_path: str,
    output_plot_path: str,
    frequency_hz: float,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    spike_times, psp_data, column_names = load_experimental_data(input_csv_path)

    noise_df = pd.read_csv(noise_fit_csv_path)
    noise_models = noise_df["model"].tolist()
    noise_params = noise_df["param"].tolist()
    if len(noise_models) != len(spike_times):
        raise ValueError("Noise fit rows do not match spike time count")

    results = []
    all_samples_list = []

    total = len(column_names)
    if progress_callback is not None:
        progress_callback(0, total, "")
    for col_idx, name in enumerate(column_names):
        print(f"Sampling {col_idx + 1}/{total}")
        data = psp_data[:, col_idx]
        all_samples = run_mcmc_chains(data, spike_times, frequency_hz, noise_models, noise_params)
        all_samples_list.append(all_samples)
        map_theta, map_post = find_map_estimate(all_samples, data, spike_times, frequency_hz, noise_models, noise_params)

        results.append({
            "Experiment": name,
            "U": map_theta[2],
            "D": map_theta[0],
            "F": map_theta[1],
            "f": map_theta[3],
            "MAP": map_post,
        })
        if progress_callback is not None:
            progress_callback(col_idx + 1, total, name)

    ensure_dir(os.path.dirname(output_csv_path))
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False, float_format="%.6f")

    ensure_dir(os.path.dirname(output_plot_path))
    plot_posterior_distributions(all_samples_list, output_plot_path)

    return results_df
