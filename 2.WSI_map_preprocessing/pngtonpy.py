import os
import cv2
import numpy as np


# =========================================================
# Convert PNG images to NumPy (.npy) files
# =========================================================

def png_to_npy(input_folder, output_folder):

    png_files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith(".png")
    ])

    print(f"Total PNG files found: {len(png_files)}")

    for filename in png_files:

        input_path = os.path.join(input_folder, filename)

        # Load PNG image
        image = cv2.imread(
            input_path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            print(f"Could not read image: {filename}")
            continue

        # Save as .npy file
        output_path = os.path.join(
            output_folder,
            filename[:-4] + ".npy"
        )

        np.save(output_path, image)

        print(f"Saved: {output_path}")


if __name__ == "__main__":

    # =====================================================
    # User-defined paths
    # =====================================================

    input_folder = "/path/to/png_directory/"
    # Directory containing PNG images

    output_folder = "/path/to/output_npy_directory/"
    # Directory where .npy files will be saved

    os.makedirs(output_folder, exist_ok=True)

    png_to_npy(
        input_folder=input_folder,
        output_folder=output_folder
    )
