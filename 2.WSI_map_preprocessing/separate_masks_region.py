import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from skimage import io, color, morphology
from skimage.measure import label, regionprops
from PIL import Image
from skimage.metrics import hausdorff_distance
from skimage import io
from skimage.measure import label
from skimage.color import label2rgb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import shutil

import glob

%config Completer.use_jedi = False

import warnings
warnings.filterwarnings('ignore')

def get_files(path, rule=".png"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all

def get_bbox(mask, value):
    y,x = np.where(mask==value)
    return [min(y), max(y), min(x), max(x), max(y)-min(y), max(x) - min(x)]

def tissue_seperator(img_dir,
                     range_n_clusters = [2, 3, 4, 5, 6],
                     best_clusters = 0,
                     previous_silh_avg = 0.0):
    img = io.imread(img_dir)
    limg = label(img)
    
    if len(np.unique(limg)) ==3:
        pass
    
    else:
        y_centroids, x_centroids, data = [], [], []
        for obj in np.unique(limg)[1:]:
            coords = get_bbox(limg, obj)
            y_center = coords[1] - coords[-2]//2
            x_center = coords[3] - coords[-1]//2

            y_centroids.append(y_center)
            x_centroids.append(x_center)
            data.append([y_center,x_center])

        dataToFit = data  
        for n_clusters in range_n_clusters:
            try:
                clusterer = KMeans(n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(dataToFit)
                silhouette_avg = silhouette_score(dataToFit, cluster_labels)
                if silhouette_avg > previous_silh_avg:
                    previous_silh_avg = silhouette_avg
                    best_clusters = n_clusters
            except:
                pass

        kmeans = KMeans(n_clusters=best_clusters, random_state=0).fit(dataToFit)
        class_label = kmeans.fit_predict(dataToFit)
        class_label = (np.array(class_label)+1)*len(np.unique(limg))
        for obj_idx, obj in enumerate(np.unique(limg)[1:]):

            limg[limg==obj] = class_label[obj_idx]
    
    return img, limg

seg_list =get_files('')

for i, sub in enumerate(seg_list):
    try:
        img, limg = tissue_seperator(sub)
    except Exception as e:
        print('*' * 10)
        print(f'ERROR: idx: {i}, path: {sub}')
        print(f'Exception: {e}')
        print('*' * 10)
        img, limg = io.imread(sub), label(io.imread(sub)) 

    unique_labels_count = len(np.unique(limg)[1:])
#     print(f'Image idx: {i}, path: {sub}, unique_labels_count: {unique_labels_count}')

    if unique_labels_count == 1:
        # Destination path for images with 1 unique label
        destination_folder = ''
        os.makedirs(destination_folder, exist_ok=True)
    elif unique_labels_count == 2:
        # Destination path for images with 2 unique labels
        destination_folder = ''
        os.makedirs(destination_folder, exist_ok=True)
    elif unique_labels_count == 3:
        # Destination path for images with 3 unique labels
        destination_folder = ''
        os.makedirs(destination_folder, exist_ok=True)
    else:
        continue  # Skip if there are no matching conditions

    # Generate the filename dynamically
    original_filename = os.path.basename(seg_list[i]).split('_')[0] + '_map_256.png'
    original_image_path = os.path.join('', original_filename)
    destination_path = os.path.join(destination_folder, original_filename)

    try:
        shutil.copy(original_image_path, destination_path)
        print(f"Image copied successfully to {destination_folder}.")
    except Exception as e:
        print(f"Failed to copy image to {destination_folder}: {e}")