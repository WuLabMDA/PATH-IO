import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def get_files(path, rule=".npy"):
    files = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if file_path.endswith(rule):
                files.append(file_path)

    return sorted(files)


def png_to_npy(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    png_files = sorted([
        filename for filename in os.listdir(input_folder)
        if filename.endswith(".png")
    ])

    for filename in png_files:

        image = cv2.imread(
            os.path.join(input_folder, filename),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            print(f"Could not read image: {filename}")
            continue

        output_path = os.path.join(
            output_folder,
            filename[:-4] + ".npy"
        )

        np.save(output_path, image)

    print(f"Converted {len(png_files)} PNG files to NPY.")


def custom_crop_and_stack(
    image,
    scales,
    final_height=350,
    final_width=350,
    output_size=224
):

    pad_height = max(final_height - image.shape[0], 0)
    pad_width = max(final_width - image.shape[1], 0)

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    padded_image = np.pad(
        image,
        ((top_pad, bottom_pad), (left_pad, right_pad)),
        mode="constant",
        constant_values=0
    )

    cropped_images = []

    for scale in scales:

        crop_height = min(padded_image.shape[0], scale)
        crop_width = min(padded_image.shape[1], scale)

        y1 = (padded_image.shape[0] - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (padded_image.shape[1] - crop_width) // 2
        x2 = x1 + crop_width

        cropped_image = padded_image[y1:y2, x1:x2]

        resized_image = cv2.resize(
            cropped_image,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR
        )

        cropped_images.append(resized_image)

    stacked_image = np.stack(cropped_images, axis=2)

    return stacked_image


def generate_multiscale_stacks(
    input_npy_folder,
    output_stack_folder,
    crop_scales=[50, 100, 150]
):

    os.makedirs(output_stack_folder, exist_ok=True)

    seg_list = get_files(input_npy_folder, rule=".npy")

    print(f"Total NPY files found: {len(seg_list)}")

    for i in range(len(seg_list)):

        seg = np.load(seg_list[i])

        image_pil = Image.fromarray(np.uint8(seg))

        grayscale_image = image_pil.convert("L")

        grayscale_array = np.array(grayscale_image)

        stacked_image = custom_crop_and_stack(
            grayscale_array,
            crop_scales
        )

        filename = os.path.basename(seg_list[i])

        save_path = os.path.join(
            output_stack_folder,
            filename
        )

        np.save(save_path, stacked_image)

    print(f"Saved multiscale stacked arrays to: {output_stack_folder}")


def visualize_stack(npy_path):

    stacked_image = np.load(npy_path)

    grayscale_images = np.split(
        stacked_image,
        stacked_image.shape[2],
        axis=2
    )

    fig, axs = plt.subplots(
        1,
        stacked_image.shape[2],
        figsize=(12, 4)
    )

    for i, img in enumerate(grayscale_images):
        axs[i].imshow(img[:, :, 0], cmap="gray")
        axs[i].axis("on")

    plt.show()


if __name__ == "__main__":

    # =====================================================
    # User-defined paths
    # =====================================================

    input_png_folder = "/path/to/input_png_folder/"
    # Directory containing cropped ROI PNG images

    output_npy_folder = "/path/to/output_npy_folder/"
    # Directory where PNG images will be converted to .npy files

    output_stack_folder = "/path/to/output_multiscale_stack_folder/"
    # Directory where multiscale stacked .npy arrays will be saved

    # =====================================================
    # Step 1: Convert PNG to NPY
    # =====================================================

    png_to_npy(
        input_folder=input_png_folder,
        output_folder=output_npy_folder
    )

    # =====================================================
    # Step 2: Generate multiscale stacked ROI arrays
    # =====================================================

    generate_multiscale_stacks(
        input_npy_folder=output_npy_folder,
        output_stack_folder=output_stack_folder,
        crop_scales=[50, 100, 150]
    )

    # Optional visualization
    # visualize_stack("/path/to/output_multiscale_stack_folder/example.npy")
