import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

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
        data = data_array.flatten()
        # data = data[data != 0]
        all_images[subj][folder] = (data, data_array.shape)

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
        ax.set_ylim(0, 2500000)

plt.tight_layout()
plt.show()