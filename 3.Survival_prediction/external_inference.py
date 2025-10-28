import torch
import pickle
from Prognosis.Networks.Macro_networks_1 import resnext50_32x4d,wide_resnet50_2,resnet50,resnet18
from Prognosis.Networks.mod_resnet import ModifiedResNet,CustomMobileNetV2
from Prognosis.Networks.densenet_arch import densenet121, densenet201,regularize_path_weights#import corresponding prognostic network
import torchvision.models as models
import torch.nn as nn
# from monai.networks.nets import EfficientNet
from vit_pytorch.regionvit import RegionViT
# from mlp_mixer_pytorch import MLPMixer


from Prognosis.Networks.senet import SENet154,SEResNext50
# from Prognosis.Networks.xception import Xception3d,regularize_path_weights
# from monai.networks.nets import EfficientNet

from Prognosis.pred_v4 import test
# from Prognosis.pred_v4d_mtl import test

import random
import pandas as pd
import os

def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all



seg_list =get_files('')
# Load the CSV file that contains split information (e.g., 'Train', 'Valid', 'Test') and 'ID' columns
csv_file_path = ''

data_info = pd.read_csv(csv_file_path)

# Define the data splits you want to fetch

test_split = 'Test'

# Fetch the IDs corresponding to each data split

test_ids = data_info[data_info['Group'] == test_split]['Filename'].tolist()

seg_file_ids = [os.path.basename(seg_path) for seg_path in seg_list]
# Filter seg_list based on the fetched file IDs for each split

test_data = [seg for seg, seg_id in zip(seg_list, seg_file_ids) if seg_id in test_ids]
# Define the device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the saved pickle file

saved_pickle_path = ''
checkpoint = torch.load(saved_pickle_path)

# Create an instance of your model
# model = resnext50_32x4d()  # Replace with your model's class

model = resnet18() 


# Load the model's state dictionary from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])


# Move the model to the device
model = model.to(device)

# Put the model in evaluation mode
model.eval()

loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data)

print(cindex_test)



risk_pred_all_t = pred_test[0]
risk_pred_all_list_t = risk_pred_all_t.tolist()
filenames_list_t = [os.path.basename(path) for path in test_data]

# Create a DataFrame
df_t = pd.DataFrame({'Filename': filenames_list_t, 'Risk_Prediction': risk_pred_all_list_t})

# Save the DataFrame to a CSV file
df_t.to_csv('.csv', index=False)



