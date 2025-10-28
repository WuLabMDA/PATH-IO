import os
import cv2
import numpy as np

# Set the path to your folder containing PNG images
input_folder = ''


# Create an output folder if it doesn't exist

output_folder = '//'

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

