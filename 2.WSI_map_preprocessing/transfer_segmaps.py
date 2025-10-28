import os
import shutil

# Define the source directory where the folders containing PNG files are located
source_directory = ''

# Define the destination directory where you want to copy the PNG files
destination_directory = '/'
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Loop through folders in the source directory
for folder_name in os.listdir(source_directory):
    folder_path = os.path.join(source_directory, folder_name)

    # Check if the item in the source directory is a folder
    if os.path.isdir(folder_path):
        # Loop through files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a PNG file and contains "_map_256" in its name
            if filename.endswith('_map_256.png'):
                # Construct the full source and destination paths
                source_file_path = os.path.join(folder_path, filename)
                destination_file_path = os.path.join(destination_directory, filename)

                # Copy the PNG file to the destination directory
                shutil.copy2(source_file_path, destination_file_path)