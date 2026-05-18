import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2



INFO_PATH = "/path/to/clinical_information.csv"

HEATMAP_PATH = "/path/to/.npy"
##################################
#His-seg Heatmap Loader(MacroNet)
##################################
class SegHeatmapDatasetLoader(Dataset):
    def __init__(self, seg_filepaths, transform=None):
        super(SegHeatmapDatasetLoader, self).__init__()
        self.seg_filepaths = seg_filepaths
        self.transform = transform


    def __len__(self):
        return len(self.seg_filepaths)


    def __getitem__(self, idx):
        seg_filepath = self.seg_filepaths[idx]
        
        seg = np.load(seg_filepath)

        if self.transform is not None:
            seg = self.transform(image=seg)["image"]

        hospital = seg_filepath.split('/')[-2]  

        base_dir = INFO_PATH
        self.data = pd.read_csv('') ##clinical data file
        self.data.index = range(2, len(self.data) + 2)
      
        ID = os.path.basename(seg_filepath).split('_')[0] 
        pd_index = self.data[self.data['WSIName'].str[:-4].isin([ID])].index.values[0] 
        
        
        
        
        
        T = (self.data['OS'][pd_index])
        O = (self.data['OS_Status'][pd_index].astype(bool)).astype(int)
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)

        # T = (self.data['PFS'][pd_index])
        # O = (self.data['PFS_Status'][pd_index].astype(bool)).astype(int)
        # O = torch.tensor(O).type(torch.FloatTensor)
        # T = torch.tensor(T).type(torch.FloatTensor)
        return seg, T, O, seg_filepath


