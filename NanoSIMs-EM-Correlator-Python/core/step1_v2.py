import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Assuming these imports exist in your environment
from nanosims_reader import read_nanosims_file, min_max_normalize, apply_contrast_adjustment, \
    auto_adjust_contrast_complete


def preprocess_image(image, is_em=False, target_pixel_size=None, current_pixel_size=None):
    """
    Normalizes intensity and resizes images to a common scale for matching.
    """
    # 1. Normalize Intensity to 0-255 (Min-Max normalization)
    image = image.astype(np.float32)
    img_norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if len(img_norm.shape) == 2:
        img_norm = auto_adjust_contrast_complete(img_norm)
        img_norm = img_norm.astype(np.uint8)

    # 2. Rescale if necessary
    if is_em and target_pixel_size and current_pixel_size:
        scale_factor = current_pixel_size / target_pixel_size
        new_width = int(img_norm.shape[1] * scale_factor)
        new_height = int(img_norm.shape[0] * scale_factor)
        img_resized = cv2.resize(img_norm, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return img_resized, scale_factor

    return img_norm, 1.0


def generate_orientations(image):
    """
    Generates 8 variations of the image.
    """
    orientations = []
    base = image
    flipped = cv2.flip(image, 1)

    for img in [base, flipped]:
        for code in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            if code is None:
                rotated = img
                angle_name = "0"
            else:
                rotated = cv2.rotate(img, code)
                if code == cv2.ROTATE_90_CLOCKWISE:
                    angle_name = "90"
                elif code == cv2.ROTATE_180:
                    angle_name = "180"
                elif code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    angle_name = "270"

            is_flipped = "Flipped" if img is flipped else "Normal"
            orientations.append((rotated, f"{is_flipped}_{angle_name}"))

    return orientations


def refine_small_region(em_high_res, nano_aligned, coarse_coords, em_res_nm, nano_res_nm):
    """
    Refines alignment for the top-left (256, 256) patch of the NanoSIMS image.
    Scales EM down to match NanoSIMS for the search.
    """
    print("\n--- Refinement Step: Aligning Top-Left (256x256) ---")

    # 1. Define the specific NanoSIMS patch
    nano_patch = nano_aligned[:256, :256]
    # nano_patch = nano_aligned[-256:, -256:]

    # 2. Calculate Scale Ratio (Nano / EM)
    # Example: 97nm / 6nm = ~16.1. This is how many EM pixels represent 1 Nano pixel.
    ratio = nano_res_nm / em_res_nm

    # 3. Define Search Region in High-Res EM with Margins
    # coarse_coords = (x, y, w, h) in High-Res EM
    c_x, c_y, _, _ = coarse_coords

    # We assume the global alignment is poor, so we add a generous margin.
    # Let's search an area corresponding to the patch size + 50% margin on all sides.
    margin_nano_pixels = 128  # Margin in terms of NanoSIMS pixels
    margin_em_pixels = int(margin_nano_pixels * ratio)

    # Expected size of the 256 patch in EM pixels
    patch_span_em = int(256 * ratio)

    # Calculate crop coordinates in High-Res EM (handling boundaries)
    search_x = max(0, c_x - margin_em_pixels)
    search_y = max(0, c_y - margin_em_pixels)

    # The width/height of the search area in EM pixels
    # We need enough space for the patch + margins
    search_w = patch_span_em + (2 * margin_em_pixels)
    search_h = patch_span_em + (2 * margin_em_pixels)

    # Crop the search region from High-Res EM
    em_search_crop = em_high_res[search_y: search_y + search_h, search_x: search_x + search_w]

    if em_search_crop.size == 0:
        print("Error: Search crop is empty. Check coordinates.")
        return None, None

    # 4. Scale the EM Crop DOWN to match NanoSIMS resolution
    # We want the pixel size to be nano_res_nm.
    # Current pixel size is em_res_nm.
    # Scale factor = em_res_nm / nano_res_nm = 1 / ratio
    target_w_scaled = int(em_search_crop.shape[1] / ratio)
    target_h_scaled = int(em_search_crop.shape[0] / ratio)

    # Preprocess (Normalize + Resize)
    em_search_scaled = cv2.normalize(em_search_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    em_search_scaled = cv2.resize(em_search_scaled, (target_w_scaled, target_h_scaled), interpolation=cv2.INTER_AREA)

    # 5. Template Matching
    # Template: nano_patch (256x256)
    # Image: em_search_scaled (larger)
    if em_search_scaled.shape[0] < nano_patch.shape[0] or em_search_scaled.shape[1] < nano_patch.shape[1]:
        print("Warning: Search region smaller than template after scaling. Returning coarse crop.")
        # Fallback logic could go here
        return None, None

    res = cv2.matchTemplate(em_search_scaled, nano_patch, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print(f"Refinement Score: {max_val:.4f}")

    # 6. Map Coordinates back to High-Res EM
    # max_loc is (x, y) in the *scaled* search image
    refined_x_scaled, refined_y_scaled = max_loc

    # Convert scaled offset to High-Res offset
    offset_x_em = int(refined_x_scaled * ratio)
    offset_y_em = int(refined_y_scaled * ratio)

    # Calculate absolute coordinates in the original large EM image
    final_abs_x = search_x + offset_x_em
    final_abs_y = search_y + offset_y_em
    final_abs_w = int(256 * ratio)  # Width corresponding to 256 nano pixels
    final_abs_h = int(256 * ratio)
    print(f"Final EM Patch Coords: x={final_abs_x}, y={final_abs_y}, w={final_abs_w}, h={final_abs_h}")
    # 7. Extract Final Result
    em_final_patch = em_high_res[final_abs_y: final_abs_y + final_abs_h,
                     final_abs_x: final_abs_x + final_abs_w]

    return em_final_patch, nano_patch


def find_coarse_alignment(em_large_path, nanosims_path, em_res_nm=6, nano_res_nm=97.65625, signal='32S'):
    """
    Finds the NanoSIMS patch within the large EM map, then refines a specific sub-region.
    """
    print("--- Loading Data ---")
    em_img = cv2.imread(em_large_path, 0)
    nano_img, nano_header = read_nanosims_file(nanosims_path)

    # Calculate nano resolution if not manually passed (25um / 256px)
    if nano_res_nm is None:
        nano_res_nm = 25 * 1000 / 256
    if nano_img is None:
        return None

    nano_img = nano_img.squeeze()

    s32_pos = nano_header['Mims_mass_symbols'].split(' ').index(signal)
    print('nanosims image shape', nano_img.shape, nano_img.dtype, nano_img.min(), nano_img.max(), nano_res_nm, em_res_nm)
    if len(nano_img.shape) == 3:
        nano_img = nano_img[:, :, s32_pos]
    else:
        nano_img = nano_img[:, :, :, s32_pos]
        # nano_img = np.sum(nano_img, axis=2)  # Sum across channels if needed

        nano_img = nano_img[:, :, 1].astype(np.float32)
        # nano_img = np.sum(nano_img[:, :], axis=-1).astype(np.float32)
        print('mean min max', nano_img.mean(), nano_img.min(), nano_img.max())



    if 'MC3 spleen' in em_large_path:
        nano_img = cv2.flip(nano_img, 0)
        nano_img = nano_img[:-256, :-256]

    if em_img is None or nano_img is None:
        raise ValueError("Could not load images. Check paths.")

    # --- 1. Match Scales (Global) ---
    print("--- Rescaling EM to match NanoSIMS resolution (Global) ---")
    em_scaled, scale_factor = preprocess_image(em_img, is_em=True,
                                               target_pixel_size=nano_res_nm,
                                               current_pixel_size=em_res_nm)

    nano_norm, _ = preprocess_image(nano_img)
    nano_norm = 255 - nano_norm

    if em_scaled.shape[0] < nano_norm.shape[0] or em_scaled.shape[1] < nano_norm.shape[1]:
        em_scaled = cv2.resize(em_scaled, (nano_norm.shape[1], nano_norm.shape[0]), interpolation=cv2.INTER_AREA)
    # --- 2. Search All Orientations ---
    print("--- Searching for optimal orientation ---")
    best_score = -1
    best_location = None
    best_orientation_img = None
    best_params = ""
    multi_orientations = True
    if multi_orientations:
        nano_variations = generate_orientations(nano_norm)
    else:
        # nano_norm = np.rot90(nano_norm, k=1)
        # nano_variations = [(nano_norm, "Normal_0")]
        nano_norm = np.rot90(nano_norm, k=3)
        # nano_norm = np.flip(nano_norm, axis=1)
        nano_variations = [(nano_norm, "Normal_270")]
        # nano_norm = np.rot90(nano_norm, k=3)
        # nano_variations = [(nano_norm, "Normal_270")]

    for i, (nano_var, label) in enumerate(nano_variations):
        if em_scaled.shape[0] < nano_var.shape[0] or em_scaled.shape[1] < nano_var.shape[1]:
            continue
        res = cv2.matchTemplate(em_scaled, nano_var, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_location = max_loc
            best_orientation_img = nano_var
            best_params = label

    print(f"✅ Global Match Found! Score: {best_score:.4f}, Orientation: {best_params}")

    # --- 3. Coarse Coordinates ---
    top_left_scaled = best_location
    h_scaled, w_scaled = best_orientation_img.shape
    x_start_orig = int(top_left_scaled[0] / scale_factor)
    y_start_orig = int(top_left_scaled[1] / scale_factor)
    w_orig = int(w_scaled / scale_factor)
    h_orig = int(h_scaled / scale_factor)

    coarse_coords = (x_start_orig, y_start_orig, w_orig, h_orig)
    print('coarse coords', coarse_coords)
    # --- 4. Refine Small Region (:256, :256) ---
    # This function handles the logic of scaling EM instead of NanoSIMS
    if best_orientation_img.shape[0] > 256 or best_orientation_img.shape[1] > 256:
        em_refined, nano_refined = refine_small_region(
            em_img,
            best_orientation_img,
            coarse_coords,
            em_res_nm,
            nano_res_nm
        )
    else:
        em_refined = em_img[y_start_orig: y_start_orig + h_orig, x_start_orig: x_start_orig + w_orig]
        nano_refined = best_orientation_img

    # --- 5. Visualization (Optional) ---
    if em_refined is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Refined EM Patch")
        plt.imshow(em_refined, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Target Nano Patch (:256,:256)")
        plt.imshow(nano_refined, cmap='gray')
        plt.savefig("refined_alignment_result.png")
        # plt.show()

    return {
        "em_high_res": em_img,
        "em_refined_patch": em_refined,  # The aligned high-res EM patch
        "corrected_nanosims": nano_refined,  # The (:256, :256) nanosims patch
          "crop_coords": (x_start_orig, y_start_orig, w_orig, h_orig),
        "orientation_label": best_params
    }
