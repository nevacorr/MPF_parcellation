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

        # ------------------------- Load NIfTI -------------------------
        img_nib = nib.load(mpf_path)
        mpf_data = img_nib.get_fdata()
        dx, dy, dz = img_nib.header.get_zooms()[:3]
        nx, ny, nz = mpf_data.shape

        # ------------------------- Compute physical sizes -------------------------
        x_size = nx * dx
        y_size = ny * dy
        z_size = nz * dz

        # Maximum field-of-view for black boxes
        max_fov = max(x_size, y_size, z_size)

        # ------------------------- Plot 3-view figure -------------------------
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, (view, idx) in zip(axes, fixed_slices.items()):

            # ------------------------- Extract slice and voxel info -------------------------
            if view == "coronal":
                img_slice = mpf_data[:, idx, :].T
                mask_r = right_mask[:, idx, :].T
                mask_l = left_mask[:, idx, :].T
                voxel_x, voxel_y = dx, dz
                slice_x, slice_y = x_size, z_size
                scale_factor = 1.0
                y_shift = 0.0
            elif view == "sagittal":
                img_slice = mpf_data[idx, :, :].T
                mask_r = right_mask[idx, :, :].T
                mask_l = left_mask[idx, :, :].T
                voxel_x, voxel_y = dy, dz
                slice_x, slice_y = y_size, z_size
                scale_factor = 1.0
                y_shift = 0.0
            else:  # axial
                img_slice = mpf_data[:, :, idx].T
                mask_r = right_mask[:, :, idx].T
                mask_l = left_mask[:, :, idx].T
                voxel_x, voxel_y = dx, dy
                slice_x, slice_y = x_size, y_size
                scale_factor = 1.0  # zoom out axial slice
                y_shift = -5.0  # move down in mm (adjust as needed)

            # ------------------------- Center slice in black box -------------------------
            slice_x_scaled = slice_x * scale_factor
            slice_y_scaled = slice_y * scale_factor
            x_offset = (max_fov - slice_x_scaled) / 2
            y_offset = (max_fov - slice_y_scaled) / 2 + y_shift

            # Draw black background box (same size for all subplots)
            ax.imshow(np.zeros((2, 2)), cmap='gray', origin='lower',
                      extent=[0, max_fov, 0, max_fov], interpolation='nearest')

            # Draw brain slice with correct voxel aspect ratio
            extent = [x_offset, x_offset + slice_x_scaled,
                      y_offset, y_offset + slice_y_scaled]
            ax.imshow(img_slice, cmap='gray', origin='lower',
                      interpolation='nearest', extent=extent,
                      aspect=voxel_y / voxel_x)

            # ------------------------- Overlay contours -------------------------
            if view == "axial":
                # scale and shift contours for axial slice
                for contour in measure.find_contours(mask_r, 0.5):
                    x_mm = contour[:, 1] * voxel_x * scale_factor + x_offset
                    y_mm = contour[:, 0] * voxel_y * scale_factor + y_offset
                    ax.plot(x_mm, y_mm, 'r', linewidth=1)
                for contour in measure.find_contours(mask_l, 0.5):
                    x_mm = contour[:, 1] * voxel_x * scale_factor + x_offset
                    y_mm = contour[:, 0] * voxel_y * scale_factor + y_offset
                    ax.plot(x_mm, y_mm, 'b', linewidth=1)
            else:
                # coronal and sagittal contours remain full size
                for contour in measure.find_contours(mask_r, 0.5):
                    x_mm = contour[:, 1] * voxel_x + x_offset
                    y_mm = contour[:, 0] * voxel_y + y_offset
                    ax.plot(x_mm, y_mm, 'r', linewidth=1)
                for contour in measure.find_contours(mask_l, 0.5):
                    x_mm = contour[:, 1] * voxel_x + x_offset
                    y_mm = contour[:, 0] * voxel_y + y_offset
                    ax.plot(x_mm, y_mm, 'b', linewidth=1)

            # ------------------------- Set axis properties -------------------------
            ax.set_xlim(0, max_fov)
            ax.set_ylim(0, max_fov)
            ax.set_aspect('equal')
            ax.axis('off')

        # ------------------------- Adjust spacing -------------------------
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
        plt.show()

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