"""Run Tutorial 4 (Wishlist) with the notebook's acquisition settings.

Saves Number Obtained and Posterior Jaccard Index after each repeat.
"""

import json
import sys
from pathlib import Path

sys.path = [p for p in sys.path if "medical-chatbot" not in p]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.acquisition import run_acquisition
from src.algorithms import Wishlist
from src.metrics import get_jaccard_posterior, get_n_obtained
from src.models import MGPR, fit_matern_hypers
from src.plotting import plot_final_metrics
from src.utils import load_ternary_data, random_sampling_no_replace

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    X, Y = load_ternary_data(ROOT / "datasets" / "FeCoNi_benchmark_dataset_220501a.mat")
    n_features, n_properties = X.shape[1], Y.shape[1]
    print(f"design space {X.shape}, properties {Y.shape}", flush=True)

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    scalers = [x_scaler, y_scaler]
    X_norm = x_scaler.fit_transform(X)
    Y_norm = y_scaler.fit_transform(Y)

    algorithm = Wishlist(
        user_algo_params={
            "scalers": scalers,
            "threshold_bounds_list": [
                [[0.0, 2.0], [0.00, 0.2]],
                [[8.0, 10.0], [0.0, 0.1]],
                [[5.0, 8.0], [0.3, 0.5]],
            ],
        }
    )
    all_ids = list(np.arange(0, len(X_norm)))
    true_target_ids = algorithm.identify_subspace(f_x=Y_norm, x=X_norm)
    print(f"wishlist targets {len(true_target_ids)}", flush=True)

    kernel_initial = (
        ConstantKernel(constant_value=1.0, constant_value_bounds=[0.01, 3.0])
        * Matern(
            nu=5 / 2,
            length_scale=n_features * [1.0],
            length_scale_bounds=n_features * [[0.01, 3.0]],
        )
        + WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
    )
    kernel_initial_list = n_properties * [kernel_initial]

    n_posterior_samples = 15
    n_initial = 1
    n_iters = 250
    n_repeats = 2
    fixed_hypers = True

    metrics_path = OUT / "tutorial4_wishlist_metrics.json"
    if metrics_path.exists() and "--only" in sys.argv:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    else:
        metrics = {
            "SwitchBAX": {"n_obtained": [], "jaccard_posterior_index": [], "switch_strategy": []},
            "US": {"n_obtained": [], "jaccard_posterior_index": [], "switch_strategy": []},
            "MeanBAX": {"n_obtained": [], "jaccard_posterior_index": [], "switch_strategy": []},
            "InfoBAX": {"n_obtained": [], "jaccard_posterior_index": [], "switch_strategy": []},
        }
    random_sampling = [
        random_sampling_no_replace(len(X_norm), len(true_target_ids), n)
        for n in range(n_initial, n_iters)
    ]
    if n_iters <= len(true_target_ids):
        best_possible_n_obtained = np.arange(n_initial, n_iters + n_initial)
    else:
        best_possible_n_obtained = list(np.arange(n_initial, len(true_target_ids))) + list(
            len(true_target_ids) * np.ones(n_iters + n_initial - len(true_target_ids))
        )

    print("fitting GP hyperparameters on the full dataset", flush=True)
    kernel_list = fit_matern_hypers(
        x_train=X_norm,
        y_train=Y_norm,
        kernel_list=kernel_initial_list,
        n_restarts_optimizer=1,
    )

    strategies = ["SwitchBAX", "MeanBAX", "US", "InfoBAX"]
    if "--only" in sys.argv:
        strategies = [sys.argv[sys.argv.index("--only") + 1]]
    for strategy in strategies:
        for j in range(n_repeats):
            print(f"Using strategy: {strategy}, repeat number: {j}", flush=True)
            np.random.seed(j)
            train_indices = list(np.random.choice(all_ids, n_initial))
            x_train = X_norm[train_indices]
            y_train = Y_norm[train_indices]
            collected_ids = list(train_indices)
            n_obtained_list = []
            jaccard_posterior_list = []
            switch_list = []

            for i in tqdm(range(n_iters)):
                multi_gpr = MGPR(kernel_list=kernel_list)
                x_train, y_train, model, collected_ids, acquisition_function, switch_strategy = run_acquisition(
                    x_train,
                    y_train,
                    X_norm,
                    Y_norm,
                    strategy,
                    algorithm,
                    multi_gpr,
                    collected_ids,
                    n_posterior_samples,
                )
                posterior_mean, _posterior_std = model.predict(X_norm)
                predicted_target_ids = algorithm.identify_subspace(f_x=posterior_mean, x=X_norm)
                n_obtained_list.append(get_n_obtained(collected_ids, true_target_ids))
                jaccard_posterior_list.append(get_jaccard_posterior(predicted_target_ids, true_target_ids))
                switch_list.append(switch_strategy)

            metrics[strategy]["n_obtained"].append(n_obtained_list)
            metrics[strategy]["jaccard_posterior_index"].append(jaccard_posterior_list)
            metrics[strategy]["switch_strategy"].append(switch_list)
            save_metrics(metrics, n_iters, len(true_target_ids))
            final_n = n_obtained_list[-1]
            final_j = jaccard_posterior_list[-1]
            print(f"finished {strategy} repeat {j}: n_obtained={final_n}, jaccard={final_j:.4f}", flush=True)

    finished = [name for name in ["SwitchBAX", "MeanBAX", "US", "InfoBAX"] if metrics[name]["n_obtained"]]
    plot_final_metrics(n_iters, metrics, finished, best_possible_n_obtained, random_sampling)
    fig = plt.gcf()
    fig.savefig(OUT / "tutorial4_wishlist_metrics.png", dpi=150)
    print(f"saved {OUT / 'tutorial4_wishlist_metrics.png'}", flush=True)


def save_metrics(metrics, n_iters, n_targets):
    payload = {
        "n_iters": n_iters,
        "n_targets": n_targets,
        "goal": "Wishlist",
        "dataset": "FeCoNi_benchmark_dataset_220501a.mat",
        "metrics": metrics,
    }
    path = OUT / "tutorial4_wishlist_metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
