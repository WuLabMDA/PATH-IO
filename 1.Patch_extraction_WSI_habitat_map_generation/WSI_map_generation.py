import os, sys
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


classes = ['Background','Bronchi', 'Inflammatory', 'Lung', 'Necrosis', 'Stroma', 'Tumor', 'Vessel']
label_color_dict = {0: (54, 54, 54), 1: (251, 217, 204), 2: (41, 150, 191), 3: (1, 0, 128), 
                    4: (142, 142, 142), 5: (71, 169, 45), 6: (229, 76, 37),7: (187, 47, 154)}
slide_root_dir = '' ## path of the prediction directory containg csv files
slide_list = sorted([ele for ele in os.listdir(slide_root_dir) if os.path.isdir(os.path.join(slide_root_dir, ele))])
for idx, cur_slide in enumerate(slide_list):
    cur_slide_name = os.path.splitext(cur_slide)[0]
    slide_pred_path = os.path.join(slide_root_dir,cur_slide, "{}_prediction_256.csv".format(cur_slide_name))
    pred_df = pd.read_csv(slide_pred_path)
    patch_fnames = pred_df["fname"].tolist()
    patch_labels = pred_df["pred"].tolist()
    demo_fname = os.path.splitext(patch_fnames[0])[0]
    underline_indices = [i for i in range(len(demo_fname)) if demo_fname.startswith("_", i)]
    map_width = int(demo_fname[underline_indices[3]+1:underline_indices[4]]) + 1
    map_height = int(demo_fname[underline_indices[6]+1:]) + 1
    wsi_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    for cur_fname, label in zip(patch_fnames, patch_labels):
        underline_indices = [i for i in range(len(cur_fname)) if cur_fname.startswith("_", i)]
        x_loc = int(cur_fname[underline_indices[2]+1:underline_indices[3]])
        y_loc = int(cur_fname[underline_indices[5]+1:underline_indices[6]])
        wsi_map[y_loc, x_loc,:] = label_color_dict[label]
    slide_map_path = os.path.join(slide_root_dir,cur_slide, "{}_map_256.png".format(cur_slide_name))
    io.imsave(slide_map_path, wsi_map)


