import os
import cv2
import numpy as np

from PIL import Image
from skimage.measure import label, regionprops


# =========================================================
# Tissue grayscale categories
# =========================================================

grayscale_categories = [0, 15, 101, 117, 122, 126, 142, 226]

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


# =========================================================
# Get files
# =========================================================

def get_files(path, rule=".png"):

    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:

            file_path = os.path.join(root, filename)

            if file_path.endswith(rule):
                files.append(file_path)

    return sorted(files)


# =========================================================
# Convert grayscale map to RGB map
# =========================================================

def grayscale_to_rgb(grayscale_image):

    colored_image_array = np.zeros(
        (grayscale_image.shape[0],
         grayscale_image.shape[1],
         3),
        dtype=np.uint8
    )

    for y in range(grayscale_image.shape[0]):
        for x in range(grayscale_image.shape[1]):

            grayscale_pixel = grayscale_image[y, x]

            if grayscale_pixel in grayscale_categories:

                grayscale_category = grayscale_to_color_category[
                    grayscale_pixel
                ]

                color_pixel = color_category_to_color[
                    grayscale_category
                ]

                colored_image_array[y, x] = color_pixel

    return Image.fromarray(colored_image_array, 'RGB')


# =========================================================
# Generate filtered tissue mask
# =========================================================

def generate_filtered_mask(segmentation_map,
                           threshold_value=54,
                           min_object_size=10):

    thresholded_image = np.where(
        segmentation_map == threshold_value,
        0,
        255
    ).astype(np.uint8)

    height, width = thresholded_image.shape

    modified_image = thresholded_image.copy()

    modified_image[:, width - 1] = 0
    modified_image[height - 1, :] = 0

    labeled_image = label(modified_image)

    regions = regionprops(labeled_image)

    filtered_image = np.zeros_like(modified_image)

    for region in regions:
        if region.area >= min_object_size:
            filtered_image[labeled_image == region.label] = 1

    return filtered_image


# =========================================================
# Crop foreground region
# =========================================================

def crop_foreground(segmentation_map, filtered_mask):

    org = segmentation_map * filtered_mask

    coords = cv2.findNonZero(org)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    cropped_image = org[y:y+h, x:x+w]

    return cropped_image


# =========================================================
# Split image into N ROIs
# =========================================================

def split_image(cropped_image, n_rois):

    cropped_images = []

    width = cropped_image.shape[1]

    split_points = np.linspace(0, width, n_rois + 1).astype(int)

    for i in range(n_rois):

        img_part = cropped_image[
            :,
            split_points[i]:split_points[i+1]
        ]

        if np.any(img_part):

            rows = np.any(img_part, axis=1)
            cols = np.any(img_part, axis=0)

            img_crop = img_part[np.ix_(rows, cols)]

            cropped_images.append(img_crop)

    return cropped_images


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    seg_map_dir = "/path/to/segmentation_maps/"

    output_dir = "/path/to/output_crops/"

    os.makedirs(output_dir, exist_ok=True)

    seg_list = get_files(seg_map_dir)

    n_rois = 3

    min_object_size = 10

    for i in range(len(seg_list)):

        print(f"Processing: {os.path.basename(seg_list[i])}")

        segmentation_map_color = cv2.imread(seg_list[i])

        if segmentation_map_color is None:
            print("Could not read image.")
            continue

        segmentation_map = cv2.cvtColor(
            segmentation_map_color,
            cv2.COLOR_BGR2GRAY
        )

        filtered_mask = generate_filtered_mask(
            segmentation_map,
            threshold_value=54,
            min_object_size=min_object_size
        )

        cropped_image = crop_foreground(
            segmentation_map,
            filtered_mask
        )

        if cropped_image is None:
            print("No foreground found.")
            continue

        cropped_images = split_image(
            cropped_image,
            n_rois=n_rois
        )

        nm = os.path.basename(seg_list[i]).split('_')[0]

        for img_idx, crop_img in enumerate(cropped_images):

            colored_image = grayscale_to_rgb(
                np.array(crop_img)
            )

            save_crop_path = os.path.join(
                output_dir,
                f"{nm}_{img_idx}.png"
            )

            colored_image.save(save_crop_path)

        print(f"Saved {len(cropped_images)} ROI crops.")
