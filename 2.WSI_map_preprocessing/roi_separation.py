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

seg_list =get_files('')

#### Single ROI ###

for i in range (len(seg_list)):
    segmentation_map_color = cv2.imread(seg_list[i])

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
    min_object_size = 10  # Adjust this value according to your needs 50 for cptac

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
            


    org = segmentation_map*filtered_image
    coords = cv2.findNonZero(org)

    # Get the bounding box of the non-zero region
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the binary image based on the bounding box
    cropped_org_image = org[y:y+h, x:x+w]
    
    contours, _ = cv2.findContours(cropped_org_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Prepare a list to store cropped images
    cropped_images = []

    # Check if there's only one image
    if np.any(cropped_org_image):
        cropped_images.append(cropped_org_image)
    else:
        divide_line_x_left = cropped_org_image.shape[1] // 3
        divide_line_x_right = 2 * cropped_org_image.shape[1] // 3

        # Split the image into left, middle, and right parts
        left_image = cropped_org_image[:, :divide_line_x_left]
        middle_image = cropped_org_image[:, divide_line_x_left:divide_line_x_right]
        right_image = cropped_org_image[:, divide_line_x_right:]

        # Prepare a list to store cropped images
        cropped_images = []

        # Handle left image
        if np.any(left_image):
            left_rows, left_cols = np.any(left_image, axis=1), np.any(left_image, axis=0)
            left_cropped = left_image[np.ix_(left_rows, left_cols)]
            cropped_images.append(left_cropped)

        # Handle middle image
        if np.any(middle_image):
            middle_rows, middle_cols = np.any(middle_image, axis=1), np.any(middle_image, axis=0)
            middle_cropped = middle_image[np.ix_(middle_rows, middle_cols)]
            cropped_images.append(middle_cropped)

        # Handle right image
        if np.any(right_image):
            right_rows, right_cols = np.any(right_image, axis=1), np.any(right_image, axis=0)
            right_cropped = right_image[np.ix_(right_rows, right_cols)]
            cropped_images.append(right_cropped)
    
    for img in range (len(cropped_images)): 
        # Load your grayscale image as a NumPy array
        grayscale_image = np.array(cropped_images[img])  # Replace with the actual path

        # Create a new array to store the colored pixels
        colored_image_array = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8)

        # Iterate through each pixel in the grayscale image
        for y in range(grayscale_image.shape[0]):
            for x in range(grayscale_image.shape[1]):
                grayscale_pixel = grayscale_image[y, x]
                if grayscale_pixel in grayscale_categories:
                    grayscale_category = grayscale_to_color_category[grayscale_pixel]
                    color_pixel = color_category_to_color[grayscale_category]
                    colored_image_array[y, x] = color_pixel

        # Convert the colored image array to a PIL Image
        colored_image = Image.fromarray(colored_image_array, 'RGB')
        nm = os.path.basename(seg_list[i]).split('_')[0]
#         output_folder = "/Data/cptac-lusc_cropped_1/"
        output_folder = "/Data/roussy_he_cropped_1/"
        os.makedirs(output_folder, exist_ok=True)
        save_crop_path = output_folder + nm + "_"+str(img)+".png"
        colored_image.save(save_crop_path)

## Two ROIS####

for i in range (len(seg_list)):
    segmentation_map_color = cv2.imread(seg_list[i])

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
    min_object_size = 10  # Adjust this value according to your needs

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
            


    org = segmentation_map*filtered_image
    coords = cv2.findNonZero(org)

    # Get the bounding box of the non-zero region
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the binary image based on the bounding box
    cropped_org_image = org[y:y+h, x:x+w]
    
    contours, _ = cv2.findContours(cropped_org_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Prepare a list to store cropped images
    cropped_images = []
    # Calculate divide line
    divide_line_x = cropped_org_image.shape[1] // 2

    # Split the image into left and right sides
    left_side_image = cropped_org_image[:, :divide_line_x]
    right_side_image = cropped_org_image[:, divide_line_x:]

    # Create masks for both sides
    left_mask = np.ones_like(left_side_image)
    right_mask = np.ones_like(right_side_image)

    # Eliminate background area
    left_side_image = left_side_image * left_mask
    right_side_image = right_side_image * right_mask
    # Find non-zero rows and columns to get the bounding box
    left_rows, left_cols = np.any(left_side_image, axis=1), np.any(left_side_image, axis=0)
    right_rows, right_cols = np.any(right_side_image, axis=1), np.any(right_side_image, axis=0)

    # Crop the images using bounding box
    left_side_image1 = left_side_image[np.ix_(left_rows, left_cols)]
    right_side_image1 = right_side_image[np.ix_(right_rows, right_cols)]
    cropped_images.append(left_side_image1)
    cropped_images.append(right_side_image1)
    
    for img in range (len(cropped_images)): 
        # Load your grayscale image as a NumPy array
        grayscale_image = np.array(cropped_images[img])  # Replace with the actual path

        # Create a new array to store the colored pixels
        colored_image_array = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8)

        # Iterate through each pixel in the grayscale image
        for y in range(grayscale_image.shape[0]):
            for x in range(grayscale_image.shape[1]):
                grayscale_pixel = grayscale_image[y, x]
                if grayscale_pixel in grayscale_categories:
                    grayscale_category = grayscale_to_color_category[grayscale_pixel]
                    color_pixel = color_category_to_color[grayscale_category]
                    colored_image_array[y, x] = color_pixel

        # Convert the colored image array to a PIL Image
        colored_image = Image.fromarray(colored_image_array, 'RGB')
        nm = os.path.basename(seg_list[i]).split('_')[0]
#         output_folder = "/Data/cptac-lusc_cropped_2/"
        output_folder = "/Data/cimac_he_cropped_2/"
        os.makedirs(output_folder, exist_ok=True)
        save_crop_path = output_folder + nm + "_"+str(img)+".png"
        colored_image.save(save_crop_path)

### Three ROIs#####
# seg_list =get_files('/Data/new_maps3/')
for i in range (len(seg_list)):
    segmentation_map_color = cv2.imread(seg_list[i])

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
    min_object_size = 50  # Adjust this value according to your needs

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
            

    org = segmentation_map*filtered_image
    coords = cv2.findNonZero(org)
    
    # Get the bounding box of the non-zero region
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the binary image based on the bounding box
    cropped_org_image = org[y:y+h, x:x+w]
    contours, _ = cv2.findContours(cropped_org_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate divide lines for both sides
    divide_line_x_left = cropped_org_image.shape[1] // 3
    divide_line_x_right = 2 * cropped_org_image.shape[1] // 3

    # Split the image into left, middle, and right parts
    left_image = cropped_org_image[:, :divide_line_x_left]
    middle_image = cropped_org_image[:, divide_line_x_left:divide_line_x_right]
    right_image = cropped_org_image[:, divide_line_x_right:]

    # Create masks for all parts
    left_mask = np.ones_like(left_image)
    middle_mask = np.ones_like(middle_image)
    right_mask = np.ones_like(right_image)

    # Eliminate background area for all parts
    left_image = left_image * left_mask
    middle_image = middle_image * middle_mask
    right_image = right_image * right_mask


    # Prepare a list to store cropped images
    cropped_images = []

    # Handle left image
    if np.any(left_image):
        left_rows, left_cols = np.any(left_image, axis=1), np.any(left_image, axis=0)
        left_cropped = left_image[np.ix_(left_rows, left_cols)]
        cropped_images.append(left_cropped)

    # Handle middle image
    if np.any(middle_image):
        middle_rows, middle_cols = np.any(middle_image, axis=1), np.any(middle_image, axis=0)
        middle_cropped = middle_image[np.ix_(middle_rows, middle_cols)]
        cropped_images.append(middle_cropped)

    # Handle right image
    if np.any(right_image):
        right_rows, right_cols = np.any(right_image, axis=1), np.any(right_image, axis=0)
        right_cropped = right_image[np.ix_(right_rows, right_cols)]
        cropped_images.append(right_cropped)
        
    for img in range (len(cropped_images)): 
    # Load your grayscale image as a NumPy array
        grayscale_image = np.array(cropped_images[img])  # Replace with the actual path

        # Create a new array to store the colored pixels
        colored_image_array = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8)

        # Iterate through each pixel in the grayscale image
        for y in range(grayscale_image.shape[0]):
            for x in range(grayscale_image.shape[1]):
                grayscale_pixel = grayscale_image[y, x]
                if grayscale_pixel in grayscale_categories:
                    grayscale_category = grayscale_to_color_category[grayscale_pixel]
                    color_pixel = color_category_to_color[grayscale_category]
                    colored_image_array[y, x] = color_pixel
        # Convert the colored image array to a PIL Image
        colored_image = Image.fromarray(colored_image_array, 'RGB')
        nm = os.path.basename(seg_list[i]).split('_')[0]
#         output_folder = "/Data/cptac-lusc_cropped_3/"
        output_folder = "/Data/mayo_he_cropped_3/"
        os.makedirs(output_folder, exist_ok=True)
        save_crop_path = output_folder + nm + "_"+str(img)+".png"
        colored_image.save(save_crop_path)
