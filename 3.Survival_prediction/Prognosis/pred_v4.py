from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import cv2
from Prognosis.data_loaders import SegHeatmapDatasetLoader#import corresponding dataloader
# from Prognosis.data_loaders_mayo import SegHeatmapDatasetLoader
# from Prognosis.data_loaders_cimac import SegHeatmapDatasetLoader
#from Prognosis.data_loaders_roussy import SegHeatmapDatasetLoader

# from Prognosis.Networks.densenet_arch import densenet121, regularize_path_weights#import corresponding prognostic network
# from Prognosis.Networks.Macro_networks_1 import resnext50_32x4d, resnet50, resnet34, regularize_path_weights#import corresponding prognostic network
# from Prognosis.Networks.mod_resnet import regularize_path_weights#import corresponding prognostic network
# from Prognosis.Networks.monai_efficient import regularize_path_weights
# from Prognosis.Networks.densenet_weight import regularize_path_weights

from Prognosis.Networks.Macro_networks_v4 import resnext50_32x4d, resnet34, regularize_path_weights#import corresponding prognostic network
from Prognosis.utils import CoxLoss,nll_loss,ce_loss,CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import pickle

BATCH_SIZE = 32
LAMBDA_COX = 1
LAMBDA_REG = 3e-4
def test(model, data, device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")):
    model.eval()

    test_transform = A.Compose(
        [   

            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ##uncomment if augnorm
            ToTensorV2(),
        ]
    )
    custom_data_loader = SegHeatmapDatasetLoader(data, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=BATCH_SIZE, shuffle=False,drop_last=False)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (x_path, survtime, censor,_) in enumerate(test_loader):

        censor = censor.to(device)
        x_path = x_path.to(device).type(torch.FloatTensor)
        x_path = x_path.to(device) ##while prediction ##
        _, pred = model(x_path)
        # pred = model(x_path)

        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_reg = regularize_path_weights(model=model)
        loss = LAMBDA_COX*loss_cox + LAMBDA_REG*loss_reg
       
        # loss = nll_loss(pred, survtime, censor,device)
        loss_test += loss.data.item()
        gt_all = None

        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information




    # Measuring Test Loss, C-Index, P-Value

    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    grad_acc_test = None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test
