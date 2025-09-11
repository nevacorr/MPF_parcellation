import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom
from matplotlib.pyplot import tight_layout

data_dir = '/Users/nevao/Documents/MPF_Project/freesurfer_scripts/overlay_parcellations_mpf_mprage/'
imagefile = 'mri/orig/001.mgz'

subjects = ['H05-2', 'H08-2', 'H10-2']
suffixes = ['_MPFcor_freesurfer','_MPFcor_sqrt_freesurfer', '_mprage1_freesurfer']

#Load the .mgz files
all_images = {}
for subj in subjects:
    all_images[subj] = {}
    for suf in suffixes:
        folder = subj+suf
        path = os.path.join(data_dir, folder, imagefile)
        img = nib.load(path)
        data_array = img.get_fdata()
        if suf == '_mprage1_freesurfer':
            maskpath = os.path.join(data_dir, folder, 'mri/brainmask.mgz')
            maskimg = nib.load(maskpath)
            # reorient freesurfer mask to match orientation of native mprage
            mask_resampled = resample_from_to(maskimg, img, order=0) #nearest neighbor interpolation
            mask_array = mask_resampled.get_fdata()
            binarymask = mask_array > 0
            masked_image = data_array * binarymask

            # Display a slice in subplot 1,3 (Python is 0-indexed: row 0, column 2)
            z = int(2 * data_array.shape[2] / 3)  # higher slice in the volume

            # Display all three side by side
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].imshow(data_array[:, :, z], cmap='gray')
            axes[0].set_title("Unmasked")
            axes[0].axis('off')

            axes[1].imshow(binarymask[:, :, z], cmap='Reds', alpha=0.5)
            axes[1].set_title("Mask")
            axes[1].axis('off')

            axes[2].imshow(masked_image[:, :, z], cmap='gray')
            axes[2].set_title("Masked image")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

            data_array = masked_image

        # Display a middle slice in the z-axis
        z = int(2 * data_array.shape[2] / 3)
        plt.imshow(data_array[:, :, z], cmap='gray')
        plt.title(f"{folder} — slice {z}")
        plt.axis('off')
        plt.show()

        data = data_array.flatten()
        data = data[data != 0]
        all_images[subj][folder] = (data, data_array.shape)

# Define consistent bins across all histograms
all_values = []
for subj_data in all_images.values():
    for vals, shape in subj_data.values():
        all_values.extend(vals)

all_values = np.array(all_values)
global_min = all_values.min()
global_max = all_values.max()
num_bins = 40
bins = np.linspace(global_min, global_max, num_bins + 1)

# Plot histograms in a 1x3 subplot
fig, axes = plt.subplots(len(subjects), len(suffixes), figsize=(10, 8), sharey=True)

for i, subj in enumerate(subjects):
    for j, suf in enumerate(suffixes):
        folder = subj + suf
        values, shape = all_images[subj][folder]
        ax = axes[i, j]
        ax.hist(values, bins=40, color="steelblue", alpha=0.7)
        ax.set_title(f"{folder}\ndim: {shape}", fontsize=10)
        ax.set_xlabel("Voxel Value")
        if j==0:
            ax.set_ylabel("Count")
        if suf != '_MPFcor_sqrt_freesurfer':
            ax.set_xlim(0, 2000)

plt.tight_layout()
plt.show()