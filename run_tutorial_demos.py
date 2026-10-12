"""Numeric outputs from tutorials 1-3. Plots are skipped."""

import sys
from pathlib import Path

sys.path = [p for p in sys.path if "medical-chatbot" not in p]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.algorithms import SubsetAlgorithm
from src.models import MGPR
from src.utils import load_ternary_data


class LevelBandAlgorithm(SubsetAlgorithm):
    def user_algorithm(self, f_x, x):
        a = self.user_algo_params["a"]
        b = self.user_algo_params["b"]
        f_1 = f_x[:, 0]
        ids1 = set(np.where(f_1 >= a)[0])
        ids2 = set(np.where(f_1 <= b)[0])
        return list(ids1.intersection(ids2))


class PercentileIntersection2D(SubsetAlgorithm):
    def user_algorithm(self, f_x, x):
        percentile_list = self.user_algo_params["percentile_list"]
        percentile_threshold_p1 = np.percentile(f_x[:, 0], percentile_list[0])
        percentile_threshold_p2 = np.percentile(f_x[:, 1], percentile_list[1])
        ids1 = set(np.where(f_x[:, 0] >= percentile_threshold_p1)[0])
        ids2 = set(np.where(f_x[:, 1] >= percentile_threshold_p2)[0])
        return list(ids1.union(ids2))


def get_n_obtained(collected_ids, true_target_ids) -> int:
    return len(set(collected_ids).intersection(set(true_target_ids)))


def get_jaccard_posterior(predicted_target_ids, true_target_ids) -> float:
    intersection = len(set(predicted_target_ids).intersection(set(true_target_ids)))
    union = len(set(predicted_target_ids).union(set(true_target_ids)))
    return intersection / union


def main():
    np.random.seed(47)
    X, Y = load_ternary_data(ROOT / "datasets" / "FeCoNi_benchmark_dataset_220501a.mat")
    n_features, n_properties = X.shape[1], Y.shape[1]
    print(f"Size of the design space: {X.shape}, Size of the measured property space: {Y.shape}", flush=True)

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    scalers = [x_scaler, y_scaler]
    X_norm = x_scaler.fit_transform(X)
    Y_norm = y_scaler.fit_transform(Y)

    level = LevelBandAlgorithm(user_algo_params={"scalers": scalers, "a": 5.0, "b": 10.0})
    level_ids = level.identify_subspace(f_x=Y_norm, x=X_norm)
    print(f"Tutorial 1 level band 5 <= property1 <= 10: {len(level_ids)} compositions", flush=True)
    print("First 10 property rows in that band:", flush=True)
    print(Y[level_ids][0:10], flush=True)

    user_algo_params = {"scalers": scalers, "percentile_list": [90, 90]}
    algorithm = PercentileIntersection2D(user_algo_params)
    target_subset_ids = algorithm.identify_subspace(f_x=Y_norm, x=X_norm)
    print(f"Tutorial 1/3 percentile union (90, 90): {len(target_subset_ids)} compositions", flush=True)
    print(
        "List of 10 composition tuples that meet the design specification: ",
        *X[target_subset_ids][0:10],
        sep=", ",
        flush=True,
    )

    n_data = 50
    collected_ids = np.random.choice(np.arange(0, X_norm.shape[0] + 1), size=n_data, replace=False)
    n_obtained = get_n_obtained(collected_ids=collected_ids, true_target_ids=target_subset_ids)
    print(f"Tutorial 3 Number Obtained from 50 random draws: {n_obtained}", flush=True)

    X_train, _X_test, y_train, _y_test = train_test_split(X_norm, Y_norm, test_size=0.975, random_state=42)
    print(f"Tutorial 2/3 GP training set shape: {X_train.shape}, {y_train.shape}", flush=True)
    kernel_initial = (
        ConstantKernel(constant_value=1.0, constant_value_bounds=[0.01, 3.0])
        * Matern(
            nu=5 / 2,
            length_scale=n_features * [1.0],
            length_scale_bounds=n_features * [[0.01, 3.0]],
        )
        + WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
    )
    multi_gpr = MGPR(kernel_list=n_properties * [kernel_initial])
    multi_gpr.fit(X_train, y_train)
    posterior_mean, _posterior_std = multi_gpr.predict(X_norm)
    posterior_mean_ids = algorithm.identify_subspace(f_x=posterior_mean, x=X_norm)
    jaccard = get_jaccard_posterior(predicted_target_ids=posterior_mean_ids, true_target_ids=target_subset_ids)
    print(f"Tutorial 2 predicted subset size: {len(posterior_mean_ids)}", flush=True)
    print(f"Tutorial 3 Posterior Jaccard Index: {jaccard}", flush=True)


if __name__ == "__main__":
    main()
