import os
import random
import pickle
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from Prognosis.data_loaders import SegHeatmapDatasetLoader
from Prognosis.Networks.densenet_arch import densenet201, regularize_path_weights
from Prognosis.utils import (
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
    count_parameters
)


warnings.filterwarnings("ignore")


BATCH_SIZE = 64
EPOCH = 100
LR = 1e-3
LAMBDA_COX = 1
LAMBDA_REG = 3e-4


# ============================================================
# Transform
# ============================================================

additional_augmentations = [
    A.HorizontalFlip(p=0.5),  # Horizontal flip with a 50% probability
    A.RandomRotate90(p=0.5),  # Randomly rotate by 90 degrees with a 50% probability
]

train_transform = A.Compose(
    [   
     
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # *additional_augmentations,

        ToTensorV2(),
    ]
)


test_transform = A.Compose(
    [
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)


# ============================================================
# Train function
# ============================================================

def train(
    train_data,
    valid_data,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):

    print(device)

    cindex_test_max = 0

    cudnn.deterministic = True
    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = densenet201()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=4e-4
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        threshold=0.01,
        patience=5
    )

    print("Number of Trainable Parameters: %d" % count_parameters(model))

    # --------------------------------------------------------
    # Train loader
    # --------------------------------------------------------

    custom_data_loader = SegHeatmapDatasetLoader(
        train_data,
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    metric_logger = {
        "train": {
            "loss": [],
            "pvalue": [],
            "cindex": [],
            "surv_acc": [],
            "grad_acc": []
        },
        "valid": {
            "loss": [],
            "pvalue": [],
            "cindex": [],
            "surv_acc": [],
            "grad_acc": []
        }
    }

    best_cindex = 0
    best_model = None
    current_patience = 0
    patience_limit = 90
    best_epoch = 0

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    save_path = "/path/to/output_directory/"
    os.makedirs(save_path, exist_ok=True)

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in tqdm(range(EPOCH)):

        model.train()

        risk_pred_all = np.array([])
        censor_all = np.array([])
        survtime_all = np.array([])

        loss_epoch = 0
        grad_acc_epoch = None

        risk_pred_filenames = []

        for batch_idx, (x_path, survtime, censor, filenames) in enumerate(train_loader):

            filenames = [os.path.basename(file) for file in filenames]

            censor = censor.to(device)
            x_path = x_path.float().to(device)

            pred = model(x_path)

            loss_cox = CoxLoss(
                survtime,
                censor,
                pred,
                device
            )

            loss_reg = regularize_path_weights(model=model)

            loss = LAMBDA_COX * loss_cox + LAMBDA_REG * loss_reg

            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_preds = pred.detach().cpu().numpy().reshape(-1)

            risk_pred_all = np.concatenate(
                (risk_pred_all, batch_preds)
            )

            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1))
            )

            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1))
            )

            risk_pred_filenames.extend(
                list(zip(filenames, batch_preds))
            )

        # ----------------------------------------------------
        # Save risk predictions for each epoch
        # ----------------------------------------------------

        df = pd.DataFrame(
            risk_pred_filenames,
            columns=["Filename", "Risk_Prediction"]
        )

        df.to_csv(
            os.path.join(save_path, f"risk_predictions_epoch_{epoch}.csv"),
            index=False
        )

        scheduler.step(loss_epoch)

        loss_epoch /= len(train_loader.dataset)

        cindex_epoch = CIndex_lifeline(
            risk_pred_all,
            censor_all,
            survtime_all
        )

        pvalue_epoch = cox_log_rank(
            risk_pred_all,
            censor_all,
            survtime_all
        )

        surv_acc_epoch = accuracy_cox(
            risk_pred_all,
            censor_all
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(
            model,
            valid_data,
            device
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if cindex_test > best_cindex:

            best_cindex = cindex_test
            best_epoch = epoch
            current_patience = 0

            if isinstance(model, nn.DataParallel):
                best_model = model.module.state_dict()
            else:
                best_model = model.state_dict()

            best_metrics = {
                "epoch": epoch,
                "data": [train_data, valid_data],
                "model_state_dict": best_model,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": {
                    "train": {
                        "loss": loss_epoch,
                        "cindex": cindex_epoch,
                        "pvalue": pvalue_epoch,
                    },
                    "valid": {
                        "loss": loss_test,
                        "cindex": cindex_test,
                        "pvalue": pvalue_test,
                    },
                },
            }

            torch.save(
                best_metrics,
                os.path.join(save_path, f"best_model_{epoch}.pkl")
            )

        else:
            current_patience += 1

        # ----------------------------------------------------
        # Metric logger
        # ----------------------------------------------------

        metric_logger["train"]["loss"].append(loss_epoch)
        metric_logger["train"]["cindex"].append(cindex_epoch)
        metric_logger["train"]["pvalue"].append(pvalue_epoch)
        metric_logger["train"]["surv_acc"].append(surv_acc_epoch)
        metric_logger["train"]["grad_acc"].append(grad_acc_epoch)

        metric_logger["valid"]["loss"].append(loss_test)
        metric_logger["valid"]["cindex"].append(cindex_test)
        metric_logger["valid"]["pvalue"].append(pvalue_test)
        metric_logger["valid"]["surv_acc"].append(surv_acc_test)
        metric_logger["valid"]["grad_acc"].append(grad_acc_test)

        print(f"Epoch {epoch + 1}/{EPOCH}")
        print(
            f"Train Loss: {loss_epoch:.4f}, "
            f"C-Index: {cindex_epoch:.4f}, "
            f"p-value: {pvalue_epoch:.4f}"
        )
        print(
            f"Valid Loss: {loss_test:.4f}, "
            f"C-Index: {cindex_test:.4f}, "
            f"p-value: {pvalue_test:.4f}\n"
        )

        # ----------------------------------------------------
        # Save model at every epoch
        # ----------------------------------------------------

        if cindex_test_max < cindex_test:
            cindex_test_max = cindex_test

        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "data": [train_data, valid_data],
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metric_logger
            },
            os.path.join(save_path, f"{epoch}.pkl")
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if current_patience >= patience_limit:
            print(
                f"Early stopping at epoch {epoch}, "
                f"best C-Index: {best_cindex:.4f}"
            )
            break

    # --------------------------------------------------------
    # Save final best model
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": best_epoch,
            "epoch_all": epoch,
            "data": [train_data, valid_data],
            "model_state_dict": best_model,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_cindex": best_cindex,
            "metric_logger": metric_logger,
        },
        os.path.join(save_path, f"best_{best_epoch}.pkl")
    )

    pickle.dump(
        pred_test,
        open(os.path.join(save_path, "pred_test.pkl"), "wb")
    )

    print(
        f"Final Train Loss: {loss_epoch:.4f}, "
        f"Final C-Index: {cindex_epoch:.4f}, "
        f"Final p-value: {pvalue_epoch:.4f}"
    )

    print(
        f"Final Valid Loss: {loss_test:.4f}, "
        f"Final C-Index: {cindex_test:.4f}, "
        f"Final p-value: {pvalue_test:.4f}"
    )

    return model, optimizer, metric_logger


# ============================================================
# Test / validation function
# ============================================================

def test(
    model,
    data,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):

    model.eval()

    custom_data_loader = SegHeatmapDatasetLoader(
        data,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    risk_pred_all = np.array([])
    censor_all = np.array([])
    survtime_all = np.array([])

    probs_all = None
    gt_all = np.array([])

    loss_test = 0
    grad_acc_test = None

    with torch.no_grad():

        for batch_idx, (x_path, survtime, censor, filenames) in enumerate(test_loader):

            censor = censor.to(device)
            x_path = x_path.float().to(device)

            pred = model(x_path)

            loss_cox = CoxLoss(
                survtime,
                censor,
                pred,
                device
            )

            loss_reg = regularize_path_weights(model=model)

            loss = LAMBDA_COX * loss_cox + LAMBDA_REG * loss_reg

            loss_test += loss.item()

            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1))
            )

            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1))
            )

            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1))
            )

    loss_test /= len(test_loader.dataset)

    cindex_test = CIndex_lifeline(
        risk_pred_all,
        censor_all,
        survtime_all
    )

    pvalue_test = cox_log_rank(
        risk_pred_all,
        censor_all,
        survtime_all
    )

    surv_acc_test = accuracy_cox(
        risk_pred_all,
        censor_all
    )

    pred_test = [
        risk_pred_all,
        survtime_all,
        censor_all,
        probs_all,
        gt_all
    ]

    return (
        loss_test,
        cindex_test,
        pvalue_test,
        surv_acc_test,
        grad_acc_test,
        pred_test
    )




    train(
        train_data=train_data,
        valid_data=valid_data
    )
