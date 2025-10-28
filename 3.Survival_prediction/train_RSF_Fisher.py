from sklearn.model_selection import KFold,StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest

import numpy as np
import pandas as pd
  # Import any relevant metrics
import pathlib
import os, sys
import warnings
warnings.simplefilter(action='ignore')
import scipy.io as sio
from numpy import unique

import torchtuples as tt
import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import KMeans


import random
import numpy as np
import torch

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sksurv.ensemble import RandomSurvivalForest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def set_all_seeds(seed):
    """
    Sets random seed for all major libraries to ensure reproducibility.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables non-deterministic optimizations

# Example usage
set_all_seeds(42)
 
def train_gmm(descriptors, n_components):
  """
  Train a Gaussian Mixture Model (GMM) for Fisher Vector encoding.

  Parameters:
      descriptors (ndarray): Local descriptors from training images.
      n_components (int): Number of Gaussian components.

  Returns:
      gmm (GaussianMixture): Trained GMM.
  """
  gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=42)
  gmm.fit(descriptors)
  return gmm

def fisher_vector_encoding(descriptors, gmm):
  """
  Perform Fisher Vector encoding on a set of local descriptors.

  Parameters:
      descriptors (ndarray): Local descriptors of an image.
      gmm (GaussianMixture): Pre-trained GMM.

  Returns:
      fisher_vector (ndarray): Fisher Vector of fixed length.
  """
  n_components = gmm.n_components
  n_features = descriptors.shape[1]

  # GMM parameters
  weights = gmm.weights_
  means = gmm.means_
  covariances = gmm.covariances_

  # Posterior probabilities (responsibilities)
  posteriors = gmm.predict_proba(descriptors)  # Shape: (n_descriptors, n_components)

  # Fisher vector components
  fisher_vector = []

  for k in range(n_components):
      # Compute first-order statistics
      diff = descriptors - means[k]
      scaled_diff = posteriors[:, k][:, np.newaxis] * diff
      fisher_vector.append(np.sum(scaled_diff, axis=0) / np.sqrt(weights[k]))

      # Compute second-order statistics
      squared_diff = posteriors[:, k][:, np.newaxis] * (diff ** 2 - covariances[k])
      fisher_vector.append(np.sum(squared_diff, axis=0) / np.sqrt(2 * weights[k]))

  fisher_vector = np.concatenate(fisher_vector)

  # Power normalization
  fisher_vector = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector))
  # L2 normalization
  fisher_vector /= np.sqrt(np.sum(fisher_vector ** 2) + 1e-10)

  return fisher_vector

df_train = pd.read_csv("").drop_duplicates('case_ID').reset_index(drop=True)
df_valid = pd.read_csv("").drop_duplicates('case_ID').reset_index(drop=True)
df_test = pd.read_csv("").drop_duplicates('case_ID').reset_index(drop=True) 
df_combined = pd.concat([df_train, df_valid], ignore_index=True)



all_risks = pd.read_excel(".xlsx")
for i in range(len(all_risks['Filename'])):
    file = all_risks['Filename'][i].split('-', 1)
#         all_risks['Filename'][i] = file[0]
    all_risks['Filename'] = all_risks['Filename'].apply(lambda x: x.split('-', 1)[0])



train_descriptors_list = []
for case_id in df_combined['case_ID']:
    index = all_risks[all_risks['Filename'] == str(case_id)].drop(columns=['Filename']).values
    train_descriptors_list.append(index)
    
    
test_descriptors_list = []
for case_id in df_test['case_ID']:
    index = all_risks[all_risks['Filename'] == str(case_id)].drop(columns=['Filename']).values
    test_descriptors_list.append(index)  
    
    
 
  
    
all_train_descriptors = np.vstack(train_descriptors_list)


print("train_descriptors_list  =  ", all_train_descriptors.shape)


def train_and_evaluate_model(train_vlad_vectors, test_vlad_vectors_list, df_combined, test_cohort_list, random_state=42):
    # Extract target variables (OS_Status and OS) from the training dataset
    y_train = df_combined[['OS_Status', 'OS']]
    
    # Convert y_train to structured array
    events_train = y_train['OS_Status'].astype(bool).tolist()
    time_values_train = y_train['OS'].astype(float).tolist()
    y_train_structured = np.array(list(zip(events_train, time_values_train)), dtype=[('event', bool), ('time', float)])

    # Train the RandomSurvivalForest using VLAD vectors as input features
    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=10,
        max_depth=2,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )

    # Train the model on the VLAD vectors from the training set
    rsf.fit(train_vlad_vectors, y_train_structured)

    # Initialize a dictionary to store results for each cohort
    results = {}

    # Loop over the test cohorts and corresponding test VLAD vectors
    for cohort_name, (test_vlad_vectors, test_cohort) in zip(['default'], zip(test_vlad_vectors_list, test_cohort_list)):
        # Extract target variables (OS_Status and OS) from the test cohort
        y_test = test_cohort[['OS_Status', 'OS']]
        
        # Convert y_test to structured array
        events_test = y_test['OS_Status'].astype(bool).tolist()
        time_values_test = y_test['OS'].astype(float).tolist()
        y_test_structured = np.array(list(zip(events_test, time_values_test)), dtype=[('event', bool), ('time', float)])

        # Evaluate the model on the test cohort
        test_c_index = rsf.score(test_vlad_vectors, y_test_structured)
        test_risks = rsf.predict(test_vlad_vectors)
        train_risks = rsf.predict(train_vlad_vectors)
        train_c_index = rsf.score(train_vlad_vectors, y_train_structured)

        # Store the results in the dictionary
        results[cohort_name] = {
            'test_c_index': test_c_index,
            'train_c_index': train_c_index,
            'test_risks': test_risks,
            'train_risks': train_risks,
        }

    return results

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Hyperparameters

n_clusters_list = [5, 6, 8, 10, 12, 14, 15, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60] #added 12
pca_energy_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


is_normalize = True

# Placeholder for results
results_list = []

for n_clusters in n_clusters_list:
    print(f"Processing n_clusters = {n_clusters}")
    
    # Train GMM model
    gmm_model = train_gmm(all_train_descriptors, n_clusters)
    
    for pca_energy in pca_energy_list:
        print(f"  Processing pca_energy = {pca_energy}")
        
        # Initialize PCA and scaler
        pca = PCA(n_components=pca_energy, svd_solver='full')
        scaler = MinMaxScaler()
        
        # Training Fisher vectors
        train_fisher_vectors = []
        for descriptors in train_descriptors_list:
            fisher_vector = fisher_vector_encoding(descriptors, gmm_model).reshape(1, -1)
            train_fisher_vectors.append(fisher_vector)
        
        normalized_train_fisher_vector = np.vstack(train_fisher_vectors)
        if pca_energy > 0:
            pca.fit(normalized_train_fisher_vector)
            normalized_train_fisher_vector = pca.transform(normalized_train_fisher_vector)
        
        if is_normalize:
            normalized_train_fisher_vector = scaler.fit_transform(normalized_train_fisher_vector)
        
        # Testing Fisher vectors for all cohorts
        test_fisher_vectors_list = []
        for test_descriptors in [test_descriptors_list, test_descriptors_list_cimac, test_descriptors_list_mayo,test_descriptors_list_tcga,test_descriptors_list_cptac,test_descriptors_list_roussy]:
            cohort_fisher_vectors = []
            for descriptors in test_descriptors:
                fisher_vector = fisher_vector_encoding(descriptors, gmm_model).reshape(1, -1)
                cohort_fisher_vectors.append(fisher_vector)
            
            normalized_test_fisher_vector = np.vstack(cohort_fisher_vectors)
            if pca_energy > 0:
                normalized_test_fisher_vector = pca.transform(normalized_test_fisher_vector)
            if is_normalize:
                normalized_test_fisher_vector = scaler.transform(normalized_test_fisher_vector)
            
            test_fisher_vectors_list.append(normalized_test_fisher_vector)
        
        # Train and evaluate the model
        results = train_and_evaluate_model(
            normalized_train_fisher_vector, test_fisher_vectors_list, df_combined, 
            [df_test, df_test_cimac, df_test_mayo,df_test_tcga,df_test_cptac,df_test_roussy]
        )
        
        # Collect results
        for cohort_name, result in results.items():
            results_list.append({
                'n_clusters': n_clusters,
                'pca_energy': pca_energy,
                'cohort': cohort_name,
                'test_c_index': result['test_c_index'],
                'train_c_index': result['train_c_index']
            })

# # # Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv("", index=False)
print("Results saved to fisher_results.csv")


train_risks_default = results['default']['train_risks']
test_risks_default = results['default']['test_risks']

test_risk = test_risks_default


# 
train_risk = train_risks_default
test_risks = pd.DataFrame(test_risk,columns=['Risk'])
train_risks = pd.DataFrame(train_risk,columns=['Risk'])

df_test_risk_default = pd.concat([df_test, test_risks], axis=1)
df_test_risk_default.to_csv('test_risk_default_fisher_os_new.csv')

df_disc_risk = pd.concat([df_combined, train_risks], axis=1)
df_disc_risk.to_csv('disc_risk_fisher_os_new.csv')