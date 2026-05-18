import os
import shutil


if __name__ == "__main__":

    # =====================================================
    # User-defined paths
    # =====================================================

    source_directory = "/path/to/source_directory/"
    # Directory containing slide-wise folders

    destination_directory = "/path/to/output_directory/"
    # Directory where all *_map_256.png files will be copied

    os.makedirs(destination_directory, exist_ok=True)

    total_copied = 0

    # =====================================================
    # Copy tissue habitat maps
    # =====================================================

    for folder_name in os.listdir(source_directory):

        folder_path = os.path.join(
            source_directory,
            folder_name
        )

        # Check if item is a directory
        if os.path.isdir(folder_path):

            for filename in os.listdir(folder_path):

                # Copy only tissue habitat maps
                if filename.endswith("_map_256.png"):

                    source_file_path = os.path.join(
                        folder_path,
                        filename
                    )

                    destination_file_path = os.path.join(
                        destination_directory,
                        filename
                    )

                    shutil.copy2(
                        source_file_path,
                        destination_file_path
                    )

                    total_copied += 1

                    print(f"Copied: {filename}")

    print(f"\nTotal files copied: {total_copied}")
