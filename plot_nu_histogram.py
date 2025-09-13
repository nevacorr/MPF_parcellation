import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom
from matplotlib.pyplot import tight_layout

data_dir = '/Users/nevao/Documents/MPF_Project/freesurfer_scripts/overlay_parcellations_mpf_mprage/'
imagefile = 'mri/nu.mgz'
maskfile = 'mri/brainmask.mgz'

subjects = ['H05-2', 'H08-2', 'H10-2']
suffixes = ['_MPFcor_freesurfer','_MPFcor_sqrt_freesurfer', '_mprage1_freesurfer']

#Load the .mgz files
all_images = {}
all_values = []
for subj in subjects:
    all_images[subj] = {}
    for suf in suffixes:
        folder = subj+suf
        nu_path = os.path.join(data_dir, folder, imagefile)
        mask_path = os.path.join(data_dir, folder, maskfile)

        nu_img = nib.load(nu_path)
        mask_img = nib.load(mask_path)

        nu_data = nu_img.get_fdata()
        mask_data = mask_img.get_fdata()

        # apply brainmask: zero outside brain
        nu_masked = nu_data.copy()
        nu_masked[mask_data == 0] = 0

        brain_voxels = nu_masked[nu_masked > 0]

        # store values and original shape
        all_images[subj][folder] = (brain_voxels, nu_data.shape)

        # add to global list for bin calculation
        all_values.extend(brain_voxels)


# Define consistent bins across all histograms
all_values = np.array(all_values)
global_min = all_values.min()
global_max = all_values.max()
num_bins = 40
bins = np.linspace(global_min, global_max, num_bins + 1)

# Plot histograms in a 1x3 subplot
fig, axes = plt.subplots(len(subjects), len(suffixes), figsize=(10, 8), sharey=False)

for i, subj in enumerate(subjects):
    for j, suf in enumerate(suffixes):
        folder = subj + suf
        values, shape = all_images[subj][folder]
        ax = axes[i, j]
        ax.hist(values, bins=bins, color="steelblue", alpha=0.7)
        ax.set_title(f"nu image for {folder}\ndim: {shape}", fontsize=10)
        ax.set_xlabel("Voxel Value")
        if j==0:
            ax.set_ylabel("Count")

plt.tight_layout()
plt.show()