import os
import cv2
import numpy as np

# Set the path to your folder containing PNG images

input_folder = ''

# Create an output folder if it doesn't exist

output_folder = ''
os.makedirs(output_folder, exist_ok=True)

# Function to convert PNG images to NumPy files
def png_to_npy(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Load the PNG image using OpenCV
            image = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_UNCHANGED)

            # Save the image as a NumPy array in .npy format
            np.save(os.path.join(output_folder, filename[:-4] + '.npy'), image)

if __name__ == "__main__":
    png_to_npy(input_folder, output_folder)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from skimage import io, color, morphology
from skimage.measure import label, regionprops
from PIL import Image

def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all

import numpy as np
import cv2

def custom_crop_and_stack(image, scales, final_height=350, final_width=350):
    # Calculate the padding needed to achieve the desired final dimensions
    pad_height = max(final_height - image.shape[0], 0)
    pad_width = max(final_width - image.shape[1], 0)

    # Calculate padding on top, bottom, left, and right to keep the image centered
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Pad the input image while keeping it centered
    padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)

    cropped_images = []

    for scale in scales:
        # Calculate the crop size
        crop_height = min(padded_image.shape[0], scale)
        crop_width = min(padded_image.shape[1], scale)

        # Calculate crop coordinates for center cropping
        y1 = (padded_image.shape[0] - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (padded_image.shape[1] - crop_width) // 2
        x2 = x1 + crop_width

        # Crop the padded image
        cropped_image = padded_image[y1:y2, x1:x2]

#         Resize the cropped image to a consistent size (e.g., 200x200)
        resized_image = cv2.resize(cropped_image, (224, 224),interpolation=cv2.INTER_LINEAR)

        # Append the resized image to the list
        cropped_images.append(resized_image)

    # Stack the cropped images along the third axis (z-direction)
    stacked_image = np.stack(cropped_images, axis=2)

    return stacked_image


seg_list =get_files('')


crop_scales = [50,100,150]

for i in range (len(seg_list)):
    seg = np.load(seg_list[i])
    image_pil = Image.fromarray(np.uint8(seg))

    # Convert the Pillow Image to grayscale
    grayscale_image = image_pil.convert('L')

    # Convert the grayscale Pillow Image back to a numpy array
    grayscale_array = np.array(grayscale_image)
    
    image = custom_crop_and_stack(grayscale_array, crop_scales)
    nm = os.path.basename(seg_list[i])
    ## Define the path to be saved ##
    save_path = "" 

    os.makedirs(save_path, exist_ok=True)
    save_path_nm = save_path + nm 
    np.save(save_path_nm,image)


### Visualization ####
import numpy as np
st = np.load('')

import numpy as np
import matplotlib.pyplot as plt

# Split the stacked image into three grayscale images
grayscale_images = np.split(st, 3, axis=2)

# Create subplots to display each image
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Display each grayscale image
for i, img in enumerate(grayscale_images):
    axs[i].imshow(img[:, :, 0], cmap='gray')  # Assuming you want to display the first channel as grayscale
#     axs[i].set_title(f'Image {i+1}')
    axs[i].axis('on')

plt.show()