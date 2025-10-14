import nibabel as nib
import numpy as np
import os
import pandas as pd

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

# Prepare list for results
results = []

# Loop over all MPF images
for mpf_file in os.listdir(mpf_folder):
    if mpf_file.endswith('.nii.gz'):
        mpf_path = os.path.join(mpf_folder, mpf_file)
        mpf_img = nib.load(mpf_path)
        mpf_data = mpf_img.get_fdata()

        # Compute mean, min, max values inside masks
        right_vals = mpf_data[right_mask]
        left_vals = mpf_data[left_mask]

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

# Convert to DataFrame
df = pd.DataFrame(results)

# Save as CSV
df.to_csv(os.path.join(register_folder, 'MPF_stats_in_S-MATT_roi_lt_rt_by_scan.csv'), index=False)

print("Done!")