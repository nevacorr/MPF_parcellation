import nibabel as nib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import binary_erosion
import csv

# Folder containing MPF images
register_folder = '/Users/nevao/Documents/MPF_Project/freesurfer_scripts/register_to_SMATT_template'
mpf_folder = os.path.join(register_folder, 'regmpf2mni_output/MPF_in_MNIadj_space_all_subj')

# Paths to your masks
right_mask_file = os.path.join(register_folder, 'SMATT_roi_right.nii.gz')
left_mask_file = os.path.join(register_folder, 'SMATT_roi_left.nii.gz')

# Load masks
right_mask = nib.load(right_mask_file).get_fdata().astype(bool)
left_mask = nib.load(left_mask_file).get_fdata().astype(bool)

# Squeeze in case images have singleton dimensions
right_mask = np.squeeze(right_mask)
left_mask = np.squeeze(left_mask)

# Erode masks by 1 voxel
structure = np.ones((3,3,3))  # 3x3x3 neighborhood
right_mask_eroded = binary_erosion(right_mask, structure=structure, iterations=1)
left_mask_eroded = binary_erosion(left_mask, structure=structure, iterations=1)

# Prepare list for results
results = []

# Coronal slice to visualize
coronal_slice = 108

# Loop over all MPF images
for mpf_file in os.listdir(mpf_folder):
    if mpf_file.endswith('.nii.gz'):
        mpf_path = os.path.join(mpf_folder, mpf_file)
        mpf_img = nib.load(mpf_path)
        mpf_data = mpf_img.get_fdata()

        # Compute mean, min, max values inside masks
        right_vals = mpf_data[right_mask_eroded]
        left_vals = mpf_data[left_mask_eroded]

        # Extract subject ID by removing suffix
        subject_id = mpf_file.replace('_reg_to_mni_adj.nii.gz','')

        results.append({
            'Subject': subject_id,
            'MPF_Right_Mean': right_vals.mean(),
            'MPF_Right_Min': right_vals.min(),
            'MPF_Right_Max': right_vals.max(),
            'MPF_Left_Mean': left_vals.mean(),
            'MPF_Left_Min': left_vals.min(),
            'MPF_Left_Max': left_vals.max()
        })

        # --- Visualization of coronal slice ---

        y = 108
        slice_mask = right_mask_eroded[:, y, :]  # shape (X, Z)

        # Find contours
        contours = measure.find_contours(slice_mask.T, 0.5)  # transpose so axes match matplotlib display

        plt.figure(figsize=(8, 8))
        plt.imshow(mpf_data[:, y, :].T, cmap='gray', origin='lower')  # underlying image

        # Plot right mask contours in red
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

        # Repeat for left mask
        slice_mask_left = left_mask_eroded[:, y, :]
        contours_left = measure.find_contours(slice_mask_left.T, 0.5)
        for contour in contours_left:
            plt.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)

        plt.title(f"{subject_id}: Coronal slice {coronal_slice}")
        plt.axis('off')
        # plt.show()
        # --------------------------------------

# Convert to DataFrame
df = pd.DataFrame(results)

# Ensure numeric columns are floats
numeric_cols = [
    'MPF_Right_Mean', 'MPF_Right_Min', 'MPF_Right_Max',
    'MPF_Left_Mean', 'MPF_Left_Min', 'MPF_Left_Max'
]
df[numeric_cols] = df[numeric_cols].astype(float)

# Save as CSV
df.to_csv(
    os.path.join(register_folder, 'MPF_stats_in_S-MATT_roi_lt_rt_by_scan_not_eroded.csv'),
    index=False,
    float_format='%.0f',
    quoting=csv.QUOTE_MINIMAL
)
print("Done!")