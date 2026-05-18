import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from fastai.vision.all import *
from fastai.metrics import error_rate

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


tfms = aug_transforms(
    mult=3.0,
    do_flip=True,
    flip_vert=False,
    max_rotate=45.0,
    min_zoom=0.9,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.25,
    p_lighting=0.9,
    batch=False,
    min_scale=0.8
)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        fmt = '.2f'
    else:
        print('Confusion matrix, without normalization')
        fmt = 'd'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            fontsize=8,
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.tight_layout()


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    np.random.seed(42)

    train_path = Path('/path/to/training_data')   # Path to training data
    test_path = Path('/path/to/test_data')        # Path to test data

    model_save_path = '/path/to/save/path_io_resnet34.pkl'
    model_load_path = '/path/to/load/path_io_resnet34.pkl'

    class_names = [
        'Background',
        'Bronchi',
        'Inflammatory',
        'Lung',
        'Necrosis',
        'Stroma',
        'Tumor',
        'Vessel'
    ]

    data = ImageDataLoaders.from_folder(
        train_path,
        valid_pct=0.2,
        seed=99,
        batch_tfms=[Normalize.from_stats(*imagenet_stats), *tfms],
        bs=300,
        num_workers=64
    )

    data_test = ImageDataLoaders.from_folder(
        test_path,
        valid_pct=0,
        batch_tfms=Normalize.from_stats(*imagenet_stats),
        bs=64,
        drop_last=False
    )

    TRAIN = True

    if TRAIN:

        learn = vision_learner(
            data,
            resnet34,
            metrics=[error_rate, accuracy]
        )

        learn.fit_one_cycle(6, lr_max=1e-2)

        learn.unfreeze()

        learn.fit_one_cycle(4, slice(1e-6, 1e-4))

        learn.export(model_save_path)

        # =========================================================
        # Training Performance
        # =========================================================

        preds, y_gt, y_pred, losses = learn.get_preds(
            ds_idx=0,
            with_loss=True,
            with_decoded=True
        )

        train_cm = confusion_matrix(y_gt, y_pred)

        train_acc = accuracy_score(y_gt, y_pred)

        train_precision_macro = precision_score(
            y_gt,
            y_pred,
            average='macro'
        )

        train_recall_macro = recall_score(
            y_gt,
            y_pred,
            average='macro'
        )

        train_f1_macro = f1_score(
            y_gt,
            y_pred,
            average='macro'
        )

        print("\n================ TRAINING METRICS ================\n")

        print("Training Accuracy :", round(train_acc, 4))
        print("Training Macro Precision :", round(train_precision_macro, 4))
        print("Training Macro Recall :", round(train_recall_macro, 4))
        print("Training Macro F1-score :", round(train_f1_macro, 4))

        print("\nPer-class Training Metrics:\n")

        print(
            classification_report(
                y_gt,
                y_pred,
                target_names=class_names,
                digits=4
            )
        )

        # =========================================================
        # Testing Performance
        # =========================================================

        preds, y_gt, y_pred, losses = learn.get_preds(
            dl=data_test.train,
            with_loss=True,
            with_decoded=True
        )

        test_cm = confusion_matrix(y_gt, y_pred)

        test_acc = accuracy_score(y_gt, y_pred)

        test_precision_macro = precision_score(
            y_gt,
            y_pred,
            average='macro'
        )

        test_recall_macro = recall_score(
            y_gt,
            y_pred,
            average='macro'
        )

        test_f1_macro = f1_score(
            y_gt,
            y_pred,
            average='macro'
        )

        print("\n================ TESTING METRICS ================\n")

        print("Testing Accuracy :", round(test_acc, 4))
        print("Testing Macro Precision :", round(test_precision_macro, 4))
        print("Testing Macro Recall :", round(test_recall_macro, 4))
        print("Testing Macro F1-score :", round(test_f1_macro, 4))

        print("\nPer-class Testing Metrics:\n")

        print(
            classification_report(
                y_gt,
                y_pred,
                target_names=class_names,
                digits=4
            )
        )

        # =========================================================
        # Confusion Matrices
        # =========================================================

        plt.figure(figsize=(8, 8))

        plot_confusion_matrix(
            train_cm,
            classes=class_names,
            title='Confusion matrix for training'
        )

        plt.figure(figsize=(8, 8))

        plot_confusion_matrix(
            test_cm,
            classes=class_names,
            title='Confusion matrix for testing'
        )

        plt.show()

    else:

        learn = load_learner(model_load_path)

        preds, y_gt, y_pred, losses = learn.get_preds(
            dl=data_test.train,
            with_loss=True,
            with_decoded=True
        )

        test_cm = confusion_matrix(y_gt, y_pred)

        test_acc = accuracy_score(y_gt, y_pred)

        test_precision_macro = precision_score(
            y_gt,
            y_pred,
            average='macro'
        )

        test_recall_macro = recall_score(
            y_gt,
            y_pred,
            average='macro'
        )

        test_f1_macro = f1_score(
            y_gt,
            y_pred,
            average='macro'
        )

        print(test_cm)

        print("\n================ TESTING METRICS ================\n")

        print("Testing Accuracy :", round(test_acc, 4))
        print("Testing Macro Precision :", round(test_precision_macro, 4))
        print("Testing Macro Recall :", round(test_recall_macro, 4))
        print("Testing Macro F1-score :", round(test_f1_macro, 4))

        print("\nPer-class Testing Metrics:\n")

        print(
            classification_report(
                y_gt,
                y_pred,
                target_names=class_names,
                digits=4
            )
        )
