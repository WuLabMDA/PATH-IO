import os
import torch
import pandas as pd

from Prognosis.Networks.densenet_arch import densenet201
from Prognosis.pred_v4 import test


def get_files(path, rule=".npy"):
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if file_path.endswith(rule):
                files.append(file_path)

    return sorted(files)


def save_predictions(pred_output, file_list, output_csv_path):

    risk_pred_all = pred_output[0]
    risk_pred_all_list = risk_pred_all.tolist()

    filenames_list = [
        os.path.basename(path)
        for path in file_list
    ]

    df = pd.DataFrame(
        {
            "Filename": filenames_list,
            "Risk_Prediction": risk_pred_all_list
        }
    )

    df.to_csv(output_csv_path, index=False)

    print(f"Saved: {output_csv_path}")


if __name__ == "__main__":

    # =====================================================
    # User-defined paths
    # =====================================================

    seg_dir = "/path/to/npy_files/"
    # Directory containing .npy files

    csv_file_path = "/path/to/split_information.csv"
    # CSV file containing Filename and Group columns

    saved_pickle_path = "/path/to/best_model.pkl"
    # Trained model checkpoint

    output_dir = "/path/to/output_predictions/"
    # Directory where risk prediction CSV files will be saved

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # Load split information
    # =====================================================

    seg_list = get_files(seg_dir, rule=".npy")

    data_info = pd.read_csv(csv_file_path)

    train_ids = data_info[data_info["Group"] == "Train"]["Filename"].tolist()
    valid_ids = data_info[data_info["Group"] == "Valid"]["Filename"].tolist()
    test_ids = data_info[data_info["Group"] == "Test"]["Filename"].tolist()

    seg_file_ids = [
        os.path.basename(seg_path)
        for seg_path in seg_list
    ]

    train_data = [
        seg for seg, seg_id in zip(seg_list, seg_file_ids)
        if seg_id in train_ids
    ]

    valid_data = [
        seg for seg, seg_id in zip(seg_list, seg_file_ids)
        if seg_id in valid_ids
    ]

    test_data = [
        seg for seg, seg_id in zip(seg_list, seg_file_ids)
        if seg_id in test_ids
    ]

    print(f"Train files: {len(train_data)}")
    print(f"Valid files: {len(valid_data)}")
    print(f"Test files : {len(test_data)}")

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda:1" if torch.cuda.is_available() else "cpu"
    )

    # =====================================================
    # Load checkpoint
    # =====================================================

    checkpoint = torch.load(
        saved_pickle_path,
        map_location=device
    )

    # =====================================================
    # Build model
    # =====================================================

    model = densenet201()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # =====================================================
    # Evaluate train, valid, and test
    # =====================================================

    loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(
        model,
        train_data,
        device=device
    )

    loss_valid, cindex_valid, pvalue_valid, surv_acc_valid, grad_acc_valid, pred_valid = test(
        model,
        valid_data,
        device=device
    )

    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(
        model,
        test_data,
        device=device
    )

    print(f"Train C-index: {cindex_train:.4f}")
    print(f"Valid C-index: {cindex_valid:.4f}")
    print(f"Test C-index : {cindex_test:.4f}")

    # =====================================================
    # Save risk predictions
    # =====================================================

    save_predictions(
        pred_output=pred_train,
        file_list=train_data,
        output_csv_path=os.path.join(output_dir, "train_risk_predictions.csv")
    )

    save_predictions(
        pred_output=pred_valid,
        file_list=valid_data,
        output_csv_path=os.path.join(output_dir, "valid_risk_predictions.csv")
    )

    save_predictions(
        pred_output=pred_test,
        file_list=test_data,
        output_csv_path=os.path.join(output_dir, "test_risk_predictions.csv")
    )
