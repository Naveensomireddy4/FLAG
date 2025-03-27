import numpy as np
import os
# Define the folder containing the `.npz` files
source_folder = r"C:\Users\navee\IIIT-DELHI\PFLlib3\dataset\CIFAR10\task_0\train"

# Loop through all files in the folder
for file_name in os.listdir(source_folder):
    # Process only `.npz` files
    if file_name.endswith(".npz"):
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(source_folder, file_name)  # Overwrite the same file

        # Load the source file
        source_data = np.load(source_file, allow_pickle=True)

        # Extract images and labels
        images = source_data['images']  # Shape: (e.g., 89, 28, 28, 3)
        labels = source_data['labels']  # Shape: (e.g., 89,)

        # Create a dictionary with the new structure
        modified_data = {
            "x": images,  # Rename "images" to "x"
            "y": labels   # Rename "labels" to "y"
        }

        # Save the modified data to the same file
        np.savez(destination_file, data=modified_data)

        print(f"Modified file saved at: {destination_file}")
