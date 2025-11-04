import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import binary_erosion
import fnmatch

# ------------------------- Configuration -------------------------
erode_masks = 0

register_folder = '/Users/nevao/Documents/MPF_Project/freesurfer_scripts/register_to_SMATT_template'
mpf_folder = os.path.join(register_folder, 'regmpf2mni_output/MPF_in_MNIadj_space_all_subj')

right_mask_file = os.path.join(register_folder, 'SMATT_roi_right.nii.gz')
left_mask_file = os.path.join(register_folder, 'SMATT_roi_left.nii.gz')

# Output folder
fig_folder = os.path.join(
    register_folder,
    'MPF_slice_overlays_eroded_masks' if erode_masks else 'MPF_slice_overlays_noneroded_masks'
)
os.makedirs(fig_folder, exist_ok=True)

# ------------------------- Load Masks -------------------------
right_mask = np.squeeze(nib.load(right_mask_file).get_fdata().astype(bool))
left_mask = np.squeeze(nib.load(left_mask_file).get_fdata().astype(bool))

if erode_masks == 1:
    structure = np.ones((3, 3, 3))
    right_mask = binary_erosion(right_mask, structure=structure)
    left_mask = binary_erosion(left_mask, structure=structure)

# ------------------------- Interactive Viewer -------------------------
def scroll_viewer(img_data, mask_r, mask_l, view="coronal"):
    fig, ax = plt.subplots(figsize=(8, 8))

    if view == "coronal":
        max_idx = img_data.shape[1]
    elif view == "sagittal":
        max_idx = img_data.shape[0]
    else:
        max_idx = img_data.shape[2]

    idx = max_idx // 2

    def get_slice(i):
        if view == "coronal":
            return img_data[:, i, :].T, mask_r[:, i, :].T, mask_l[:, i, :].T
        elif view == "sagittal":
            return img_data[i, :, :].T, mask_r[i, :, :].T, mask_l[i, :, :].T
        else:
            return img_data[:, :, i].T, mask_r[:, :, i].T, mask_l[:, :, i].T

    def update(i):
        ax.clear()
        img, r, l = get_slice(i)
        ax.imshow(img, cmap='gray', origin='lower')
        for contour in measure.find_contours(r, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=1)
        for contour in measure.find_contours(l, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], 'b', linewidth=1)
        ax.set_title(f"{view.capitalize()} slice {i}")
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_scroll(event):
        nonlocal idx
        if event.button == 'up':
            idx = (idx + 1) % max_idx
        elif event.button == 'down':
            idx = (idx - 1) % max_idx
        update(idx)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    update(idx)
    plt.show()

fixed_slices = {
    "sagittal": 118,   # x-axis (left-right)
    "coronal": 108,   # y-axis (front-back)
    "axial": 80       # z-axis (inferior-superior)
}

# ------------------------- Process Each Subject -------------------------
for mpf_file in os.listdir(mpf_folder):
    if fnmatch.fnmatch(mpf_file, "H*reg_reg*.nii.gz"):
        mpf_path = os.path.join(mpf_folder, mpf_file)
        mpf_data = nib.load(mpf_path).get_fdata()
        subject_id = mpf_file.replace('_reg_to_mni_adj.nii.gz', '')

        print(f"Processing {subject_id}...")

        # Slice indices for each view
        slice_indices = fixed_slices

        # --- Create a single figure with 3 subplots (1x3 layout) ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax, (view, idx) in zip(axes, slice_indices.items()):
            if view == "coronal":
                img_slice = mpf_data[:, idx, :].T
                mask_r = right_mask[:, idx, :].T
                mask_l = left_mask[:, idx, :].T
            elif view == "sagittal":
                img_slice = mpf_data[idx, :, :].T
                mask_r = right_mask[idx, :, :].T
                mask_l = left_mask[idx, :, :].T
            else:  # axial
                img_slice = mpf_data[:, :, idx].T
                mask_r = right_mask[:, :, idx].T
                mask_l = left_mask[:, :, idx].T

            ax.imshow(img_slice, cmap='gray', origin='lower')
            for contour in measure.find_contours(mask_r, 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)
            for contour in measure.find_contours(mask_l, 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)

            # ax.set_title(f"{view.capitalize()} slice {idx}")
            ax.axis('off')

        # plt.suptitle(f"{subject_id} – Mask Overlays", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

        # Save one figure with all three views
        # out_path = os.path.join(fig_folder, f"{subject_id}_3view.png")
        # plt.savefig(out_path, bbox_inches='tight', dpi=200)
        # plt.close()

        # --- Optional interactive scrolling ---
        # Uncomment to scroll through each orientation interactively
        # scroll_viewer(mpf_data, right_mask, left_mask, view="coronal")
        # scroll_viewer(mpf_data, right_mask, left_mask, view="sagittal")
        # scroll_viewer(mpf_data, right_mask, left_mask, view="axial")

print("✅ All subjects processed. Combined PNGs saved to:", fig_folder)

#sag 118 ax 74 cor 115