import os
import cv2
import numpy as np

from PIL import Image
from skimage.measure import label, regionprops


def get_files(path, rule=".png"):
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if file_path.endswith(rule):
                files.append(file_path)

    return files


if __name__ == "__main__":

    # Directory containing segmentation/tissue habitat maps
    seg_map_dir = "/path/to/segmentation_map_directory/"

    # Directory where binary masks will be saved
    output_mask_dir = "/path/to/output_mask_directory/"

    os.makedirs(output_mask_dir, exist_ok=True)

    seg_list_mask = get_files(seg_map_dir, rule=".png")

    threshold_value = 54
    min_object_size = 20

    for i in range(len(seg_list_mask)):

        segmentation_map_color = cv2.imread(seg_list_mask[i])

        if segmentation_map_color is None:
            print(f"Could not read image: {seg_list_mask[i]}")
            continue

        # Convert RGB tissue map to grayscale
        segmentation_map = cv2.cvtColor(
            segmentation_map_color,
            cv2.COLOR_BGR2GRAY
        )

        # Generate binary mask
        # Background pixels with grayscale value 54 are set to 0
        # Other tissue regions are set to 255
        thresholded_image = np.where(
            segmentation_map == threshold_value,
            0,
            255
        ).astype(np.uint8)

        height, width = thresholded_image.shape

        modified_image = thresholded_image.copy()

        # Remove border artifacts
        modified_image[:, width - 1] = 0
        modified_image[height - 1, :] = 0

        # Label connected components
        labeled_image = label(modified_image)

        regions = regionprops(labeled_image)

        # Remove small objects
        filtered_image = np.zeros_like(modified_image)

        for region in regions:
            if region.area >= min_object_size:
                filtered_image[labeled_image == region.label] = 1

        scaled_img = Image.fromarray((filtered_image * 255).astype(np.uint8))

        nm = os.path.basename(seg_list_mask[i]).split("_")[0]

        save_path = os.path.join(
            output_mask_dir,
            nm + "_mask.png"
        )

        scaled_img.save(save_path)

        print(f"Saved mask: {save_path}")
