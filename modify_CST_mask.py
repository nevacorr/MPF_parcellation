import nibabel as nib
import numpy as np
import os

orig_file_dir = '~/Documents/MPF_Project/freesurfer_scripts/register_to_SMATT_template'

# Load the original mask
mask_nii = nib.load(os.path.join(orig_file_dir, 'motor_template', 'S-MATT_roi_lt_rt.nii'))
mask_data = mask_nii.get_fdata()

# Zero out voxels with Z>130
mask_data[:, :, 131:] = 0

# Save masked mask
nib.save(nib.Nifti1Image(mask_data, mask_nii.affine, mask_nii.header), 'S-MAtt_roi_lt_rt_masked.nii')

# Split mask into hemispheres
midline_x = 90

# Make right hemisphere mask
right_mask = np.zeros_like(mask_data)
right_mask[:midline_x+1, :, :] = mask_data[:midline_x+1, :, :]

# Make left hemisphere mask
left_mask = np.zeros_like(mask_data)
left_mask[midline_x+1:, :, :] = mask_data[midline_x+1:, :, :]

# Save hemispheric masks
nib.save(nib.Nifti1Image(right_mask, mask_nii.affine, mask_nii.header), os.path.join(orig_file_dir, "SMATT_roi_right.nii.gz"))
nib.save(nib.Nifti1Image(left_mask, mask_nii.affine, mask_nii.header),  os.path.join(orig_file_dir, "SMATT_roi_left.nii.gz"))


