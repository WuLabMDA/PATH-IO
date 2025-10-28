import os
import logging
import random
import cv2
# Env
# from Prognosis.data_loaders_mtl import *
# from Prognosis.train_test_mtl_v5 import train,test
from Prognosis.data_loaders import *
from Prognosis.train_test_v5 import train,test

import os
def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all
seg_list =get_files('') ## path containing the ROI npy files
import random
import pandas as pd

# Load the CSV file that contains split information (e.g., 'Train', 'Valid', 'Test') and 'ID' columns
csv_file_path = ''
data_info = pd.read_csv(csv_file_path)

# Define the data splits you want to fetch
train_split = 'Train'
valid_split = 'Valid'
test_split = 'Test'

# Fetch the IDs corresponding to each data split
train_ids = data_info[data_info['Group1'] == train_split]['Filename'].tolist()
valid_ids = data_info[data_info['Group1'] == valid_split]['Filename'].tolist()
test_ids = data_info[data_info['Group1'] == test_split]['Filename'].tolist()

seg_file_ids = [os.path.basename(seg_path) for seg_path in seg_list]
# Filter seg_list based on the fetched file IDs for each split
train_data = [seg for seg, seg_id in zip(seg_list, seg_file_ids) if seg_id in train_ids]
valid_data = [seg for seg, seg_id in zip(seg_list, seg_file_ids) if seg_id in valid_ids]
test_data = [seg for seg, seg_id in zip(seg_list, seg_file_ids) if seg_id in test_ids]

model, optimizer, metric_logger = train(train_data,valid_data)
# print(len(train_data))
# print(len(valid_data))
loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(model, train_data)
loss_valid, cindex_valid, pvalue_valid, surv_acc_valid, grad_acc_valid, pred_valid = test(model, valid_data)
loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data)

print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
print("[Final] Apply model to validation set: C-Index: %.10f, P-Value: %.10e" % (cindex_valid, pvalue_valid))
logging.info("[Final] Apply model to validation set: cC-Index: %.10f, P-Value: %.10e" % (cindex_valid, pvalue_valid))
print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))