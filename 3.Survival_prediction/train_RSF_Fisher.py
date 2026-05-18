import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sksurv.ensemble import RandomSurvivalForest


warnings.simplefilter(action="ignore")


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(42)


# ============================================================
# Train GMM
# ============================================================

def train_gmm(descriptors, n_components):

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=42
    )

    gmm.fit(descriptors)

    return gmm


# ============================================================
# Fisher vector encoding
# ============================================================

def fisher_vector_encoding(descriptors, gmm):

    n_components = gmm.n_components

    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_

    posteriors = gmm.predict_proba(descriptors)

    fisher_vector = []

    for k in range(n_components):

        diff = descriptors - means[k]

        scaled_diff = (
            posteriors[:, k][:, np.newaxis] * diff
        )

        fisher_vector.append(
            np.sum(scaled_diff, axis=0) /
            np.sqrt(weights[k])
        )

        squared_diff = (
            posteriors[:, k][:, np.newaxis] *
            (diff ** 2 - covariances[k])
        )

        fisher_vector.append(
            np.sum(squared_diff, axis=0) /
            np.sqrt(2 * weights[k])
        )

    fisher_vector = np.concatenate(fisher_vector)

    # Power normalization
    fisher_vector = (
        np.sign(fisher_vector) *
        np.sqrt(np.abs(fisher_vector))
    )

    # L2 normalization
    fisher_vector /= (
        np.sqrt(np.sum(fisher_vector ** 2) + 1e-10)
    )

    return fisher_vector


# ============================================================
# Generate Fisher vectors
# ============================================================

def generate_fisher_vectors(descriptor_list, gmm):

    fisher_vectors = []

    for descriptors in descriptor_list:

        fisher_vector = fisher_vector_encoding(
            descriptors,
            gmm
        ).reshape(1, -1)

        fisher_vectors.append(fisher_vector)

    fisher_vectors = np.vstack(fisher_vectors)

    return fisher_vectors


# ============================================================
# Train RSF
# ============================================================

def train_rsf(features, clinical_df):

    y_train = np.array(
        list(
            zip(
                clinical_df["OS_Status"].astype(bool),
                clinical_df["OS"].astype(float)
            )
        ),
        dtype=[("event", bool), ("time", float)]
    )

    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=10,
        max_depth=2,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )

    rsf.fit(features, y_train)

    return rsf


# ============================================================
# Evaluate RSF
# ============================================================

def evaluate_rsf(rsf, features, clinical_df):

    y_test = np.array(
        list(
            zip(
                clinical_df["OS_Status"].astype(bool),
                clinical_df["OS"].astype(float)
            )
        ),
        dtype=[("event", bool), ("time", float)]
    )

    c_index = rsf.score(features, y_test)

    risks = rsf.predict(features)

    return c_index, risks


# ============================================================
# User-defined paths
# ============================================================

train_csv = "/path/to/train.csv"
valid_csv = "/path/to/valid.csv"
test_csv = "/path/to/test.csv"

risk_feature_file = "/path/to/risk_features.xlsx"

results_output_csv = "/path/to/fisher_results.csv"

risk_output_dir = "/path/to/risk_outputs/"


# ============================================================
# Load cohorts
# ============================================================

df_train = pd.read_csv(train_csv).drop_duplicates("case_ID")
df_valid = pd.read_csv(valid_csv).drop_duplicates("case_ID")
df_test = pd.read_csv(test_csv).drop_duplicates("case_ID")

df_combined = pd.concat(
    [df_train, df_valid],
    ignore_index=True
)


# ============================================================
# Load ROI-level risk descriptors
# ============================================================

all_risks = pd.read_excel(risk_feature_file)

all_risks["Filename"] = (
    all_risks["Filename"]
    .apply(lambda x: x.split("-", 1)[0])
)


# ============================================================
# Build descriptor lists
# ============================================================

train_descriptors_list = []

for case_id in df_combined["case_ID"]:

    descriptors = (
        all_risks[
            all_risks["Filename"] == str(case_id)
        ]
        .drop(columns=["Filename"])
        .values
    )

    train_descriptors_list.append(descriptors)


test_descriptors_list = []

for case_id in df_test["case_ID"]:

    descriptors = (
        all_risks[
            all_risks["Filename"] == str(case_id)
        ]
        .drop(columns=["Filename"])
        .values
    )

    test_descriptors_list.append(descriptors)


all_train_descriptors = np.vstack(
    train_descriptors_list
)

print(
    "All train descriptors shape:",
    all_train_descriptors.shape
)


# ============================================================
# Hyperparameter search
# ============================================================

n_clusters_list = [5, 10, 15, 20, 25]
pca_energy_list = [0.5, 0.7, 0.9, 1.0]

results_list = []

for n_clusters in n_clusters_list:

    print(f"\nProcessing n_clusters = {n_clusters}")

    gmm_model = train_gmm(
        all_train_descriptors,
        n_clusters
    )

    for pca_energy in pca_energy_list:

        print(f"  PCA energy = {pca_energy}")

        train_fisher_vectors = generate_fisher_vectors(
            train_descriptors_list,
            gmm_model
        )

        test_fisher_vectors = generate_fisher_vectors(
            test_descriptors_list,
            gmm_model
        )

        scaler = MinMaxScaler()

        if pca_energy < 1:

            pca = PCA(
                n_components=pca_energy,
                svd_solver="full"
            )

            train_fisher_vectors = pca.fit_transform(
                train_fisher_vectors
            )

            test_fisher_vectors = pca.transform(
                test_fisher_vectors
            )

        train_fisher_vectors = scaler.fit_transform(
            train_fisher_vectors
        )

        test_fisher_vectors = scaler.transform(
            test_fisher_vectors
        )

        rsf = train_rsf(
            train_fisher_vectors,
            df_combined
        )

        train_cindex, train_risks = evaluate_rsf(
            rsf,
            train_fisher_vectors,
            df_combined
        )

        test_cindex, test_risks = evaluate_rsf(
            rsf,
            test_fisher_vectors,
            df_test
        )

        results_list.append(
            {
                "n_clusters": n_clusters,
                "pca_energy": pca_energy,
                "train_c_index": train_cindex,
                "test_c_index": test_cindex
            }
        )

        print(
            f"Train C-index: {train_cindex:.4f} | "
            f"Test C-index: {test_cindex:.4f}"
        )


# ============================================================
# Save hyperparameter search results
# ============================================================

results_df = pd.DataFrame(results_list)

results_df.to_csv(
    results_output_csv,
    index=False
)

print(f"Results saved to: {results_output_csv}")


# ============================================================
# Save final risks
# ============================================================

os.makedirs(risk_output_dir, exist_ok=True)

df_disc_risk = pd.concat(
    [
        df_combined.reset_index(drop=True),
        pd.DataFrame(train_risks, columns=["Risk"])
    ],
    axis=1
)

df_disc_risk.to_csv(
    os.path.join(
        risk_output_dir,
        "discovery_risk.csv"
    ),
    index=False
)

df_test_risk = pd.concat(
    [
        df_test.reset_index(drop=True),
        pd.DataFrame(test_risks, columns=["Risk"])
    ],
    axis=1
)

df_test_risk.to_csv(
    os.path.join(
        risk_output_dir,
        "test_risk.csv"
    ),
    index=False
)

print("Risk files saved.")
