import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from skimage import io, color, morphology
from skimage.measure import label, regionprops
from PIL import Image

def get_files(path, rule=".png"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all

# Grayscale categories in the filtered grayscale image
grayscale_categories = [0, 15, 101, 117, 122, 126, 142, 226]

# Dictionary mapping grayscale categories to color categories
grayscale_to_color_category = {
    0: 0,
    15: 3,
    101: 7,
    117: 6,
    122: 2,
    126: 5,
    142: 4,
    226: 1
}

# Dictionary mapping color categories to RGB color values
color_category_to_color = {
    0: (0, 0, 0),
    1: (251, 217, 204),
    2: (41, 150, 191),
    3: (1, 0, 128),
    4: (142, 142, 142),
    5: (71, 169, 45),
    6: (229, 76, 37),
    7: (187, 47, 154)
}
## define the directory which have the segmentation maps ##
seg_list_mask =get_files('')

for i in range (len(seg_list_mask)):
    segmentation_map_color = cv2.imread(seg_list_mask[i])

    # Convert the color image to grayscale
    segmentation_map = cv2.cvtColor(segmentation_map_color, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    threshold_value = 54
    thresholded_image = np.where(segmentation_map == threshold_value, 0, 255).astype(np.uint8)
    # Get the dimensions of the image
    height, width = thresholded_image.shape

    # Create a copy of the original image
    modified_image = thresholded_image.copy()

    # Set the rightmost column to black
    modified_image[:, width-1] = 0

    # Set the bottom row to black
    modified_image[height-1, :] = 0
    
    # Define the minimum size threshold for objects to be removed
    min_object_size = 20  # Adjust this value according to your needs

    # Label connected components in the binary image
    labeled_image = label(modified_image)

    # Calculate properties of labeled regions
    regions = regionprops(labeled_image)

    # Create an array to store the binary image after removing small objects
    filtered_image = np.zeros_like(modified_image)

    # Iterate through regions and keep only those above the threshold size
    for region in regions:
        if region.area >= min_object_size:
            filtered_image[labeled_image == region.label] = 1
            
    scaled_img = Image.fromarray(filtered_image*255)
    nm = os.path.basename(seg_list_mask[i]).split('_')[0]
    save_path = "" + nm + "_mask.png"
    scaled_img.save(save_path)
