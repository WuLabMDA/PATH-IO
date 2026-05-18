import os
import pandas as pd
import numpy as np

from skimage import io


# =========================================================
# Tissue classes
# =========================================================

classes = [
    'Background',
    'Bronchi',
    'Inflammatory',
    'Lung',
    'Necrosis',
    'Stroma',
    'Tumor',
    'Vessel'
]


# =========================================================
# RGB color map for tissue classes
# =========================================================

label_color_dict = {
    0: (54, 54, 54),      # Background
    1: (251, 217, 204),  # Bronchi
    2: (41, 150, 191),   # Inflammatory
    3: (1, 0, 128),      # Lung
    4: (142, 142, 142),  # Necrosis
    5: (71, 169, 45),    # Stroma
    6: (229, 76, 37),    # Tumor
    7: (187, 47, 154)    # Vessel
}


if __name__ == '__main__':

    # =====================================================
    # User-defined paths
    # =====================================================

    slide_root_dir = '/path/to/prediction_directory/'
    # Directory containing slide-wise prediction folders
    # Each folder should contain:
    # slide_name_prediction_256.csv

    # =====================================================
    # Get slide folders
    # =====================================================

    slide_list = sorted([
        ele for ele in os.listdir(slide_root_dir)
        if os.path.isdir(os.path.join(slide_root_dir, ele))
    ])

    # =====================================================
    # Generate tissue habitat maps
    # =====================================================

    for idx, cur_slide in enumerate(slide_list):

        cur_slide_name = os.path.splitext(cur_slide)[0]

        print(f'Processing: {cur_slide_name}')

        slide_pred_path = os.path.join(
            slide_root_dir,
            cur_slide,
            f"{cur_slide_name}_prediction_256.csv"
        )

        if not os.path.exists(slide_pred_path):
            print(f'Prediction CSV not found for {cur_slide_name}. Skipping.')
            continue

        pred_df = pd.read_csv(slide_pred_path)

        patch_fnames = pred_df["fname"].tolist()
        patch_labels = pred_df["pred"].tolist()

        if len(patch_fnames) == 0:
            print(f'No predictions found for {cur_slide_name}. Skipping.')
            continue

        # =================================================
        # Determine map size from patch filename
        # =================================================

        demo_fname = os.path.splitext(patch_fnames[0])[0]

        underline_indices = [
            i for i in range(len(demo_fname))
            if demo_fname.startswith("_", i)
        ]

        map_width = int(
            demo_fname[underline_indices[3] + 1:underline_indices[4]]
        ) + 1

        map_height = int(
            demo_fname[underline_indices[6] + 1:]
        ) + 1

        # =================================================
        # Initialize WSI map
        # =================================================

        wsi_map = np.zeros(
            (map_height, map_width, 3),
            dtype=np.uint8
        )

        # =================================================
        # Populate tissue labels
        # =================================================

        for cur_fname, label in zip(patch_fnames, patch_labels):

            underline_indices = [
                i for i in range(len(cur_fname))
                if cur_fname.startswith("_", i)
            ]

            x_loc = int(
                cur_fname[underline_indices[2] + 1:underline_indices[3]]
            )

            y_loc = int(
                cur_fname[underline_indices[5] + 1:underline_indices[6]]
            )

            wsi_map[y_loc, x_loc, :] = label_color_dict[label]

        # =================================================
        # Save tissue habitat map
        # =================================================

        slide_map_path = os.path.join(
            slide_root_dir,
            cur_slide,
            f"{cur_slide_name}_map_256.png"
        )

        io.imsave(slide_map_path, wsi_map)

        print(f'Saved tissue map to: {slide_map_path}')
