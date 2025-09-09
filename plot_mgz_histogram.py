import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.pyplot import tight_layout

data_dir = '/Users/nevao/Documents/MPF_Project/freesurfer_scripts/overlay_parcellations_mpf_mprage/'
imagefile = 'mri/orig/001.mgz'

img_dirs = [
    'H05-2_MPFcor_freesurfer',
    'H05-2_MPFcor_sqrt_freesurfer',
    'H05-2_mprage1_freesurfer'
]

#Load the .mgz files
images = {}
for d in img_dirs:
    path = os.path.join(data_dir, d, imagefile)
    img = nib.load(path)
    data=img.get_fdata().flatten()
    # data = data[data != 0]
    images[d] = data

# Plot histograms in a 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, (d, values) in zip(axes, images.items()):
    ax.hist(values, bins=100, color="steelblue", alpha=0.7)
    ax.set_title(d, fontsize=10)
    ax.set_xlabel("Voxel Value")
    ax.set_ylabel("Count")
    ax.set_ylim(0, 1000000)

plt.tight_layout()
plt.show()