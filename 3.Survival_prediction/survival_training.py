import os
import logging
import pandas as pd

from Prognosis.train_test_v5 import train, test


def get_files(path, rule=".npy"):
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if file_path.endswith(rule):
                files.append(file_path)

    return sorted(files)


if __name__ == "__main__":

    # =====================================================
    # User-defined paths
    # =====================================================

    seg_dir = "/path/to/roi_npy_files/"
    # Directory containing ROI .npy files

    csv_file_path = "/path/to/split_information.csv"
    # CSV containing Filename and Group1 columns

    # =====================================================
    # Load ROI files
    # =====================================================

    seg_list = get_files(seg_dir, rule=".npy")

    # =====================================================
    # Load split information
    # =====================================================

    data_info = pd.read_csv(csv_file_path)

    train_ids = data_info[data_info["Group1"] == "Train"]["Filename"].tolist()
    valid_ids = data_info[data_info["Group1"] == "Valid"]["Filename"].tolist()
    test_ids = data_info[data_info["Group1"] == "Test"]["Filename"].tolist()

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
    # Train model
    # =====================================================

    model, optimizer, metric_logger = train(
        train_data,
        valid_data
    )

    # =====================================================
    # Final evaluation
    # =====================================================

    loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(
        model,
        train_data
    )

    loss_valid, cindex_valid, pvalue_valid, surv_acc_valid, grad_acc_valid, pred_valid = test(
        model,
        valid_data
    )

    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(
        model,
        test_data
    )

    print(
        "[Final] Apply model to training set: "
        "C-Index: %.10f, P-Value: %.10e"
        % (cindex_train, pvalue_train)
    )

    logging.info(
        "[Final] Apply model to training set: "
        "C-Index: %.10f, P-Value: %.10e"
        % (cindex_train, pvalue_train)
    )

    print(
        "[Final] Apply model to validation set: "
        "C-Index: %.10f, P-Value: %.10e"
        % (cindex_valid, pvalue_valid)
    )

    logging.info(
        "[Final] Apply model to validation set: "
        "C-Index: %.10f, P-Value: %.10e"
        % (cindex_valid, pvalue_valid)
    )

    print(
        "[Final] Apply model to testing set: "
        "C-Index: %.10f, P-Value: %.10e"
        % (cindex_test, pvalue_test)
    )

    logging.info(
        "[Final] Apply model to testing set: "
        "C-Index: %.10f, P-Value: %.10e"
        % (cindex_test, pvalue_test)
    )
