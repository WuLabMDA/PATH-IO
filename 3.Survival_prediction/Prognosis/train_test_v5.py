import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import cv2
from Prognosis.data_loaders import SegHeatmapDatasetLoader#import corresponding dataloader
# from Prognosis.Networks.vision_transformer import ViT
# from Prognosis.Networks.vit_2d import ViT
# from vit_pytorch.regionvit import RegionViT
# from Prognosis.Networks.region_vit import RegionViT
from torchvision.models import resnet18
import torchvision.models as models
# from Prognosis.Networks.mod_resnet import CustomMobileNetV2,ModifiedResNet #regularize_path_weights#import corresponding prognostic network
# from Prognosis.Networks.Macro_networks import resnext50_32x4d, resnet18,resnet50,wide_resnet50_2, regularize_path_weights#import corresponding prognostic network

from Prognosis.Networks.densenet_arch import densenet121, densenet201, regularize_path_weights#import corresponding prognostic network
# from Prognosis.Networks.Macro_networks_1 import resnext50_32x4d, resnet18,resnet50,wide_resnet50_2, regularize_path_weights#import corresponding prognostic network
from Prognosis.Networks.monai_efficient import regularize_path_weights
from monai.networks.nets import EfficientNet

from Prognosis.Networks.senet import SENet154,SEResNext50
from Prognosis.utils import CoxLoss,nll_loss,ce_loss,CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import pickle
# from mlp_mixer_pytorch import MLPMixer


BATCH_SIZE = 64
EPOCH = 100
LR = 1e-3#1e-3 #5e-3
LAMBDA_COX = 1
LAMBDA_REG = 3e-4

best_model = None
best_cindex = 0
patience = 50  # You can adjust this value based on your preference

additional_augmentations = [
    A.HorizontalFlip(p=0.5),  # Horizontal flip with a 50% probability
    A.RandomRotate90(p=0.5),  # Randomly rotate by 90 degrees with a 50% probability
    # A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast adjustments
    # A.Blur(p=0.1),  # Apply blur with a 10% probability
    # A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), 
    # Add more augmentations as needed
]

train_transform = A.Compose(
    [   
        # A.Resize(512,512),
        # A.PadIfNeeded(min_height=200, min_width=350, border_mode=cv2.BORDER_CONSTANT, value=0),  # Zero padding min_height=200, min_width=350
        # A.CenterCrop(150, 150),  # center cropping during training ##59% with center crop (150,150) without additional augmentations
        # # A.RandomCrop(100, 100),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # *additional_augmentations,

        ToTensorV2(),
    ]
)

def train(train_data, valid_data, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print(device)
    cindex_test_max = 0
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)


    
    # efficientnet_params = {
    # # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    # "efficientnet-b0": (1.0, 1.0, 200, 0.2, 0.2),
    # "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    # "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    # "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    # "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    # "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    # "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    # "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    # "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    # "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
    # }
    # blocks_args_str = [
    #     "r1_k7_s11_e1_i32_o16_se0.25",
    #     "r2_k1_s22_e6_i16_o24_se0.25",
    #     "r2_k1_s22_e6_i24_o40_se0.25",
    #     "r3_k1_s22_e6_i40_o80_se0.25",
    #     "r3_k1_s11_e6_i80_o112_se0.25",
    #     "r4_k1_s22_e6_i112_o192_se0.25",
    #     "r1_k1_s11_e6_i192_o320_se0.25",
    #     ]
    # weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params['efficientnet-b0']
    # model = EfficientNet(
    #     blocks_args_str  = blocks_args_str,
    #     spatial_dims=2,
    #     in_channels=3,
    #     num_classes=1,
    #     width_coefficient=weight_coeff,
    #     depth_coefficient=depth_coeff,
    #     dropout_rate=dropout_rate,
    #     image_size=224,
    #     drop_connect_rate=dropconnect_rate,
    # )
    # model = SENet154(spatial_dims=2,in_channels = 3)
    # model = ViT(in_channels=3, img_size=(200,200), proj_type='conv', pos_embed_type='sincos', classification=True,
    #         spatial_dims=2,patch_size=16)
    
    # model = ViT(
    # image_size = 200,
    # patch_size = 10,
    # num_classes = 1,
    # dim = 1024,
    # depth = 6,
    # heads = 16,
    # mlp_dim = 2048,
    # dropout = 0.1,
    # emb_dropout = 0.1
    # )

    # model = RegionViT(
    # dim = (64, 128, 256, 512),      
    # depth = (2, 2, 8, 2),           
    # window_size = 7,                
    # num_classes = 1,
    # channels = 3,
    # tokenize_local_3_conv = False,  
    # use_peg = False,
    # )
    # model = MLPMixer(
    #     image_size = 224,
    #     channels = 3,
    #     patch_size = 16,
    #     dim = 512,
    #     depth = 12,
    #     num_classes = 1
    # )
    
# Create an instance of the modified ResNet model
    # pretrained_model_path = '/Data/macronet.pth'
    model = densenet201()
    # model.load_state_dict(torch.load(pretrained_model_path))
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    print("Number of Trainable Parameters: %d" % count_parameters(model))

    custom_data_loader = SegHeatmapDatasetLoader(train_data, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=BATCH_SIZE, shuffle=True,drop_last=False)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]},
                      'valid':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]}}
    best_cindex = 0  # Initialize best C-Index
    best_model = None  # Initialize best model
    current_patience = 0
    patience_limit = 90  # Set your desired patience limit here
    best_epoch = 0  # Initialize the epoch where the best C-Index is achieved
    save_path = '/Data/surv_analysis_mlp_seg_os_224/'
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    for epoch in tqdm(range(EPOCH)):

        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0
        print('train_model_before_weight')
        print(list(model.parameters())[-1])
        risk_pred_filenames = []
        for batch_idx, (x_path, survtime, censor,filenames) in enumerate(train_loader):
            filenames = [os.path.basename(file) for file in filenames]
            # print(x_path.shape)
            censor = censor.to(device)
            x_path = x_path.to(device).type(torch.FloatTensor)
            # _, pred = model(x_path)
            pred = model(x_path) ##for senet,vit##
            
            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_path_weights(model=model)
            loss = LAMBDA_COX*loss_cox + LAMBDA_REG*(loss_reg)
            # loss = nll_loss(pred, survtime, censor,device)
            # loss = ce_loss(pred, survtime, censor,device)
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
            risk_pred_filenames.extend(list(zip(filenames, risk_pred_all)))

        df = pd.DataFrame(risk_pred_filenames, columns=['Filename', 'Risk_Prediction'])
        df.to_csv(f'{save_path}risk_predictions_epoch_{epoch}.csv', index=False)

        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

        loss_epoch /= len(train_loader.dataset)
        # print(risk_pred_all)

        cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
        grad_acc_epoch = None
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, valid_data)
        
        if cindex_test > best_cindex:
            best_cindex = cindex_test
            best_model = model.state_dict()
            best_epoch = epoch
            current_patience = 0  # Reset current patience since we found a better model
            
            epoch_idx = epoch
            # save_path = '/Data/surv_analysis_v5_4d/'
            # if not os.path.exists(save_path): 
            #     os.makedirs(save_path)


             # Log the metrics for the best model
            best_metrics = {
                'epoch': epoch_idx,
                'data': [train_data, valid_data],
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'train': {
                        'loss': loss_epoch,
                        'cindex': cindex_epoch,
                        'pvalue': pvalue_epoch,
                    },
                    'test': {
                        'loss': loss_test,
                        'cindex': cindex_test,
                        'pvalue': pvalue_test,
                    },
                },
            }

            torch.save(best_metrics, f"{save_path}/best_model_{epoch_idx}.pkl")  # Save it with a specific name,

        else:
            current_patience += 1

        # Check if we have reached the patience limit
        if current_patience >= patience_limit:
            print(f"Early stopping at epoch {epoch}, best C-Index: {best_cindex}")
            break  # Early stopping criteria met, break out of the loop


        # Append the train and test losses and C-Index values to the metric_logger dictionary
        metric_logger['train']['loss'].append(loss_epoch)
        metric_logger['train']['cindex'].append(cindex_epoch)
        metric_logger['valid']['loss'].append(loss_test)
        metric_logger['valid']['cindex'].append(cindex_test)

        # Print values for each epoch
        print(f'Epoch {epoch + 1}/{EPOCH}')
        print(f'Train Loss: {loss_epoch:.4f}, C-Index: {cindex_epoch:.4f}, p-value: {pvalue_epoch:.4f}')
        print(f'Valid Loss: {loss_test:.4f}, C-Index: {cindex_test:.4f}, p-value: {pvalue_test:.4f}\n')
        ## saves all model##
        epoch_idx = epoch
        if cindex_test_max < cindex_test:
            cindex_test_max = cindex_test
        torch.save({
        'epoch':epoch_idx,
        'data': [train_data, valid_data],
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger}, 
        save_path + '/{}.pkl'.format(epoch_idx))
 

    # After the loop, save the best model if it exists
    if best_model is not None:
        epoch_idx = epoch

    # Save the best model to a file
    torch.save({
        'epoch': best_epoch,  # Save the epoch where the best C-Index is achieved
        'epoch_all': epoch_idx,
        'data': [train_data, valid_data],
        'model_state_dict': best_model,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_cindex': best_cindex,
        'metric_logger': metric_logger,
    }, f"{save_path}/best_{best_epoch}.pkl")  # Save it with a specific name


    # The final values after training
    print(f'Final Train Loss: {loss_epoch:.4f}, Final C-Index: {cindex_epoch:.4f}, Final p-value: {pvalue_epoch:.4f}')
    print(f'Final Valid Loss: {loss_test:.4f}, Final C-Index: {cindex_test:.4f}, Final p-value: {pvalue_test:.4f}')
    # print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:}'.format('Train', loss_epoch, 'C-Index', cindex_epoch, 'p-value', pvalue_epoch))
    # print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:}\n'.format('Test', loss_test, 'C-Index', cindex_test, 'p-value', pvalue_test))

    pickle.dump(pred_test, open(save_path + '/pred_test.pkl', 'wb'))

    return model, optimizer, metric_logger


def test(model, data, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()

    test_transform = A.Compose(
        [   
            # A.Resize(350,350),
            # A.PadIfNeeded(min_height=200, min_width=350, border_mode=cv2.BORDER_CONSTANT, value=0),  # Zero padding
            # A.CenterCrop(150, 150),  # Random cropping during training
            # # A.RandomCrop(100,100),
            # A.Resize(200, 200),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    custom_data_loader = SegHeatmapDatasetLoader(data, transform=train_transform)
    # custom_data_loader = SegHeatmapDatasetLoader(data, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=BATCH_SIZE, shuffle=False,drop_last=False)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (x_path, survtime, censor,_) in enumerate(test_loader):

        censor = censor.to(device)
        x_path = x_path.to(device).type(torch.FloatTensor)
        # x_path = x_path.to(device) ##while prediction ##
        # _, pred = model(x_path)
        pred = model(x_path)

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
