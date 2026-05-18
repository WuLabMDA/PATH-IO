import os
import shutil
import warnings

import numpy as np
from skimage import io
from skimage.measure import label
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


warnings.filterwarnings("ignore")


def get_files(path, rule=".png"):
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if file_path.endswith(rule):
                files.append(file_path)

    return sorted(files)


def get_bbox(mask, value):
    y, x = np.where(mask == value)

    return [
        min(y),
        max(y),
        min(x),
        max(x),
        max(y) - min(y),
        max(x) - min(x)
    ]


def tissue_separator(
    img_path,
    range_n_clusters=[2, 3, 4, 5, 6],
    best_clusters=0,
    previous_silh_avg=0.0
):
    img = io.imread(img_path)

    limg = label(img)

    if len(np.unique(limg)) == 3:
        pass

    else:
        y_centroids = []
        x_centroids = []
        data = []

        for obj in np.unique(limg)[1:]:

            coords = get_bbox(limg, obj)

            y_center = coords[1] - coords[-2] // 2
            x_center = coords[3] - coords[-1] // 2

            y_centroids.append(y_center)
            x_centroids.append(x_center)
            data.append([y_center, x_center])

        data_to_fit = data

        for n_clusters in range_n_clusters:
            try:
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=0
                )

                cluster_labels = clusterer.fit_predict(data_to_fit)

                silhouette_avg = silhouette_score(
                    data_to_fit,
                    cluster_labels
                )

                if silhouette_avg > previous_silh_avg:
                    previous_silh_avg = silhouette_avg
                    best_clusters = n_clusters

            except Exception:
                pass

        if best_clusters > 0:
            kmeans = KMeans(
                n_clusters=best_clusters,
                random_state=0
            )

            class_label = kmeans.fit_predict(data_to_fit)
            class_label = (np.array(class_label) + 1) * len(np.unique(limg))

            for obj_idx, obj in enumerate(np.unique(limg)[1:]):
                limg[limg == obj] = class_label[obj_idx]

    return img, limg


if __name__ == "__main__":

    # =====================================================
    # User-defined paths
    # =====================================================

    cropped_mask_dir = "/path/to/cropped_mask_directory/"
    # Directory containing cropped binary/ROI mask images

    original_map_dir = "/path/to/original_map_directory/"
    # Directory containing original tissue habitat maps

    output_one_roi_dir = "/path/to/output_1_roi_maps/"
    output_two_roi_dir = "/path/to/output_2_roi_maps/"
    output_three_roi_dir = "/path/to/output_3_roi_maps/"

    os.makedirs(output_one_roi_dir, exist_ok=True)
    os.makedirs(output_two_roi_dir, exist_ok=True)
    os.makedirs(output_three_roi_dir, exist_ok=True)

    seg_list = get_files(cropped_mask_dir, rule=".png")

    for i, sub in enumerate(seg_list):

        try:
            img, limg = tissue_separator(sub)

        except Exception as e:
            print("*" * 10)
            print(f"ERROR: idx: {i}, path: {sub}")
            print(f"Exception: {e}")
            print("*" * 10)

            img = io.imread(sub)
            limg = label(img)

        unique_labels_count = len(np.unique(limg)[1:])

        if unique_labels_count == 1:
            destination_folder = output_one_roi_dir

        elif unique_labels_count == 2:
            destination_folder = output_two_roi_dir

        elif unique_labels_count == 3:
            destination_folder = output_three_roi_dir

        else:
            print(
                f"Skipping {os.path.basename(sub)}: "
                f"{unique_labels_count} ROI regions detected."
            )
            continue

        original_filename = (
            os.path.basename(sub).split("_")[0] + "_map_256.png"
        )

        original_image_path = os.path.join(
            original_map_dir,
            original_filename
        )

        destination_path = os.path.join(
            destination_folder,
            original_filename
        )

        if not os.path.exists(original_image_path):
            print(f"Original map not found: {original_image_path}")
            continue

        try:
            shutil.copy(original_image_path, destination_path)

            print(
                f"Copied {original_filename} to "
                f"{destination_folder}"
            )

        except Exception as e:
            print(
                f"Failed to copy {original_filename} "
                f"to {destination_folder}: {e}"
            )
