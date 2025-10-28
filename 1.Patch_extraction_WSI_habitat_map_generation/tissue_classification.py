import os, sklearn
import glob
import numpy as np
import pdb
import matplotlib.pyplot as plt
from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
from sklearn.metrics import confusion_matrix

tfms = aug_transforms(mult=3.0, do_flip=True, flip_vert=False,
                      max_rotate=45.0, min_zoom=0.9,
                      max_zoom=1.1, max_lighting=0.2,
                      max_warp=0.2, p_affine=0.25, 
                      p_lighting=0.9, batch=False,
                      min_scale=0.8)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,fontsize=10)
    plt.yticks(tick_marks, classes,fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",fontsize=8,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.tight_layout()

if __name__ == '__main__':


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    np.random.seed(99)

    train_path = Path('') ## path to the training data

    data = ImageDataLoaders.from_folder(
                        train_path, 
                        valid_pct=0.2,
                        batch_tfms=[Normalize.from_stats(*imagenet_stats), *tfms],
                        bs=300,
                        num_workers=64)

    test_path = Path('') ## path to the test data
    test_folder = "" ##  test folder name
    data_test = ImageDataLoaders.from_folder(
                        test_path,
                        train = test_folder,
                        valid_pct=0,
                        batch_tfms=Normalize.from_stats(*imagenet_stats),
                        bs=64,
                        drop_last=False)

    TRAIN = True

    if TRAIN == True:
        learn = vision_learner(data, resnet34, metrics=[error_rate, accuracy]) #resnet34 densenet121 resnet50

        learn.fit_one_cycle(6, lr_max=1e-2)
        learn.unfreeze()

        learn.fit_one_cycle(4, slice(1e-6, 1e-4))
        learn.export('') ## path where the model to be saved

        preds, y_pred, y_gt, losses = learn.get_preds(ds_idx=0, with_loss=True, with_decoded=True)
        train_cm = confusion_matrix(y_pred, y_gt)
      
        print(sklearn.metrics.accuracy_score(y_pred, y_gt))

        preds, y_pred, y_gt, losses = learn.get_preds(dl=data_test, with_loss=True, with_decoded=True)
        test_cm = confusion_matrix(y_pred, y_gt)
        print(sklearn.metrics.accuracy_score(y_pred, y_gt))
        
        plt.figure()
        plot_confusion_matrix(train_cm, classes=['Background', 'Bronchi', 'Inflammatory', 'Lung', 'Necrosis', 'Stroma', 'Tumor', 'Vessel'], title='Confusion matrix for training')
        plt.figure()
        plot_confusion_matrix(test_cm, classes=['Background', 'Bronchi', 'Inflammatory', 'Lung', 'Necrosis', 'Stroma', 'Tumor', 'Vessel'], title='Confusion matrix for testing')
        plt.show()
    else:
        learn = vision_learner(data, models.resnet34, metrics=error_rate)
        learn = load_learner('') ## path to load the model

        preds, y_pred, y_gt, losses = learn.get_preds(dl=data_test, with_loss=True, with_decoded=True)

        print(sklearn.metrics.confusion_matrix(y_pred, y_gt))
        print(sklearn.metrics.accuracy_score(y_pred, y_gt))