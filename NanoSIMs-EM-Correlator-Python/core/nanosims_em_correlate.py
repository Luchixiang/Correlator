import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from pathlib import Path
import torch.nn.functional as F
import os
import nrrd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt

DEVICE = 'cpu'


def auto_adjust_contrast(image, auto_threshold=5000):
    if len(image.shape) != 2:
        raise ValueError("Input image must be 2D grayscale")

    if image.dtype == np.float32 or image.dtype == np.float64:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_scaled = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image_scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        image_scaled = image.astype(np.uint8) if image.max() <= 255 else (image * 255 / image.max()).astype(np.uint8)

    bins = 256
    hist_range = (0, 255)
    histogram, bin_edges = np.histogram(image_scaled.flatten(), bins=bins, range=hist_range)
    pixel_count = image.size
    bin_size = bin_edges[1] - bin_edges[0]
    limit = pixel_count // 10
    threshold = pixel_count // auto_threshold

    hmin = 0
    for i in range(256):
        if histogram[i] > limit: histogram[i] = 0
        if histogram[i] > threshold:
            hmin = i
            break

    hmax = 255
    for i in range(255, -1, -1):
        if histogram[i] > limit: histogram[i] = 0
        if histogram[i] > threshold:
            hmax = i
            break

    hist_min = bin_edges[0]
    if hmax >= hmin:
        min_val = hist_min + hmin * bin_size
        max_val = hist_min + hmax * bin_size
        if min_val == max_val:
            min_val, max_val = float(image.min()), float(image.max())
    else:
        min_val, max_val = float(image.min()), float(image.max())

    return min_val, max_val


def flow_to_affine(flow_tensor, subsample_step=5):
    """
    Convert dense optical flow to a global affine transformation matrix.

    Args:
        flow_tensor: [B, 2, H, W] tensor from RAFT
        subsample_step: Integer to subsample points for faster RANSAC computation

    Returns:
        warp_matrix: 2x3 Affine transformation matrix
    """
    # 1. Extract flow components
    # flow_tensor is [1, 2, H, W]. Convert to numpy [H, W, 2]
    print('flow_tensor', flow_tensor.shape)
    flow = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    h, w = flow.shape[:2]

    flow_x = flow[:, :, 0]  # U displacement
    flow_y = flow[:, :, 1]  # V displacement

    # 2. Create a meshgrid of original pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # 3. Calculate destination coordinates based on flow
    x_prime = x + flow_x
    y_prime = y + flow_y

    # 4. Flatten the coordinates into lists of (x, y) point pairs
    src_pts = np.column_stack((x.flatten(), y.flatten())).astype(np.float32)
    dst_pts = np.column_stack((x_prime.flatten(), y_prime.flatten())).astype(np.float32)

    # 5. Subsample points to speed up RANSAC (optional but recommended for large images)
    src_pts_sub = src_pts[::subsample_step]
    dst_pts_sub = dst_pts[::subsample_step]
    src_pts_sub = np.ascontiguousarray(src_pts_sub).reshape(-1, 1, 2)
    dst_pts_sub = np.ascontiguousarray(dst_pts_sub).reshape(-1, 1, 2)

    # 6. Estimate Affine Transform using RANSAC
    # cv2.estimateAffine2D computes full affine (translation, rotation, scale, shear)
    # If you only want rigid + scale, use cv2.estimateAffinePartial2D instead
    warp_matrix, inliers = cv2.estimateAffine2D(src_pts_sub, dst_pts_sub, method=cv2.RANSAC)

    inlier_ratio = np.sum(inliers) / len(inliers)
    print(f"Affine estimation complete. RANSAC inlier ratio: {inlier_ratio:.2%}")

    return warp_matrix


def flow_to_affine_confidence(flow_tensor, confidence_mask=None, subsample_step=5):
    """
    Convert dense optical flow to a global affine transformation matrix.

    Args:
        flow_tensor: [B, 2, H, W] tensor from RAFT
        confidence_mask: [H, W] numpy array where 1 is confident/valid, 0 is invalid
        subsample_step: Integer to subsample points for faster RANSAC computation

    Returns:
        warp_matrix: 2x3 Affine transformation matrix
    """
    # 1. Extract flow components
    flow = flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    h, w = flow.shape[:2]

    flow_x = flow[:, :, 0]  # U displacement
    flow_y = flow[:, :, 1]  # V displacement

    # 2. Create a meshgrid of original pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # 3. Calculate destination coordinates based on flow
    x_prime = x + flow_x
    y_prime = y + flow_y

    # 4. Flatten the coordinates into lists of (x, y) point pairs
    src_pts = np.column_stack((x.flatten(), y.flatten())).astype(np.float32)
    dst_pts = np.column_stack((x_prime.flatten(), y_prime.flatten())).astype(np.float32)

    # 5. FILTER BY CONFIDENCE (If provided)
    if confidence_mask is not None:
        # Ensure mask is boolean and flattened
        valid_flat = confidence_mask.flatten() > 0
        src_pts = src_pts[valid_flat]
        dst_pts = dst_pts[valid_flat]
        print(f"Filtered points using confidence mask. Kept {len(src_pts)} valid points.")

        # If we filtered out too many points, fallback to all points
        if len(src_pts) < 10:
            print("⚠️ Warning: Too few confident points. Ignoring confidence mask.")
            src_pts = np.column_stack((x.flatten(), y.flatten())).astype(np.float32)
            dst_pts = np.column_stack((x_prime.flatten(), y_prime.flatten())).astype(np.float32)

    # 6. Subsample points to speed up RANSAC
    src_pts_sub = src_pts[::subsample_step]
    dst_pts_sub = dst_pts[::subsample_step]

    # 7. Format for OpenCV: contiguous memory arrays of shape (N, 1, 2)
    src_pts_sub = np.ascontiguousarray(src_pts_sub).reshape(-1, 1, 2)
    dst_pts_sub = np.ascontiguousarray(dst_pts_sub).reshape(-1, 1, 2)

    # 8. Estimate Affine Transform using RANSAC
    warp_matrix, inliers = cv2.estimateAffine2D(src_pts_sub, dst_pts_sub, method=cv2.RANSAC)

    if warp_matrix is None:
        print("⚠️ Warning: Affine estimation failed. Returning Identity matrix.")
        warp_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    else:
        inlier_ratio = np.sum(inliers) / len(inliers)
        print(f"✅ Affine estimation complete. RANSAC inlier ratio: {inlier_ratio:.2%}")

    return warp_matrix


def apply_contrast_adjustment(image, min_val, max_val):
    if max_val <= min_val: return image
    image = image.astype(np.float32)
    adjusted = np.clip(image, min_val, max_val)
    adjusted = (adjusted - min_val) / (max_val - min_val)
    return (adjusted * 255).astype(np.uint8)


def auto_adjust_contrast_complete(image, auto_threshold=5000):
    min_val, max_val = auto_adjust_contrast(image, auto_threshold)
    return apply_contrast_adjustment(image, min_val, max_val)



######################
## Image form trans###
######################
def img2tensor(img):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


######################
# occlusion detection#
######################

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag


def compute_flow_gradients(flow):
    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    ## fw-flow: img1 => img2, tensor N,C,H,W
    ## bw-flow: img2 => img1, tensor N,C,H,W

    with torch.no_grad():
        ## warp fw-flow to img2
        fw_flow_w = warp(fw_flow, bw_flow)

        ## convert to numpy array
        fw_flow_w = tensor2img(fw_flow_w)
        fw_flow = tensor2img(fw_flow)
        bw_flow = tensor2img(bw_flow)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 1

    return occlusion


###########################
## raft functions
###########################
def load_image(imfile):
    if type(imfile) == str:
        img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    else:
        img = imfile
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_image2(img):
    img = np.stack([img, img, img], axis=2)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()
def mask_nano_boundary(img_2d, border_px=2):
    """
    Zero out `border_px` pixels on all four sides of a 2-D image.
    Returns a copy — the original is untouched.
    """
    masked = img_2d.copy()
    masked[:border_px, :]  = 0   # top
    masked[-border_px:, :] = 0   # bottom
    masked[:, :border_px]  = 0   # left
    masked[:, -border_px:] = 0   # right
    return masked


def compute_feather_mask(warped_img, feather_radius=40):
    """
    Build a smooth [0,1] alpha mask that feathers out the valid-pixel
    boundary of a warped image, eliminating hard sawtooth edges.

    Parameters
    ----------
    warped_img   : 2-D float32 array (single channel, values in [0,255])
    feather_radius : pixels over which the edge fades to zero
    """
    # 1. Binary mask: pixels that were actually filled by the warp
    valid = (warped_img > 0).astype(np.uint8)

    # 2. Morphological erosion removes isolated noise pixels at the boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    valid_clean = cv2.erode(valid, kernel, iterations=2)
    from scipy.ndimage import distance_transform_edt
    # 3. Distance transform: each pixel gets its distance to the nearest 0
    dist = distance_transform_edt(valid_clean)

    # 4. Normalise to [0,1] over the feather_radius, then clamp
    alpha = np.clip(dist / feather_radius, 0.0, 1.0).astype(np.float32)

    # 5. Smooth the ramp so the gradient itself has no staircase artefacts
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_radius * 0.3)

    return alpha  # shape (h, w), dtype float32, range [0, 1]


def demo(args, em_img, nano_img):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = model.module
    model.to(DEVICE)
    model.eval()

    em_img_original = em_img.copy()

    with torch.no_grad():
        # em_img = image_resize(em_img, width=nano_img.shape[1], height=nano_img.shape[0])
        em_img = cv2.resize(em_img, dsize=(nano_img.shape[1], nano_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        em_img = auto_adjust_contrast_complete(em_img)
        nano_img = auto_adjust_contrast_complete(nano_img)
        wf_img = nano_img
        gt_img = em_img
        print(f"EM image shape: {em_img.shape}, Nano image shape: {nano_img.shape}")
        h, w = wf_img.shape[:2]

        image1 = load_image2(gt_img)
        image2 = load_image2(wf_img)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_up_fw = model(image1, image2)
        flow_up_bw = model(image2, image1)
        occlusion_mask = detect_occlusion(flow_up_fw, flow_up_bw)

        confidence_mask = 1 - occlusion_mask
        print('confidence_map', confidence_mask.max(),
              confidence_mask.min(), confidence_mask.mean())

        warp_matrix = flow_to_affine_confidence(flow_up_fw, confidence_mask)
        image2_np = image2[0].permute(1, 2, 0).cpu().numpy()

        # ── Warp NanoSIMS → EM space ──────────────────────────────────────────
        # Use BORDER_REPLICATE instead of BORDER_CONSTANT to avoid hard zero
        # edges; the feather mask will suppress the replicated fringe anyway.
        image_warped_nano = cv2.warpAffine(
            image2_np.astype(np.float32),
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,   # ← was BORDER_CONSTANT / 0
        )  # shape: (h, w, 3)

        # ── Upsample warped nano to original EM resolution ────────────────────
        em_h, em_w = em_img_original.shape[:2]
        nano_channel = image_warped_nano[:, :, 0]  # single channel [0,255]

        # ── Build feather mask at nano resolution, then upsample ──────────────
        # We need to know which pixels were *actually* inside the original
        # NanoSIMS frame before replication, so warp a solid white canvas.
        canvas = np.ones_like(image2_np[:, :, 0], dtype=np.float32) * 255.0
        warped_canvas = cv2.warpAffine(
            canvas,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )  # 255 inside valid region, 0 outside

        feather_mask_nano = compute_feather_mask(warped_canvas, feather_radius=5)

        # Apply feathering at nano resolution before upsampling
        nano_channel_feathered = nano_channel * feather_mask_nano  # soft edges

        nano_upsampled = cv2.resize(
            nano_channel_feathered,
            (em_w, em_h),
            interpolation=cv2.INTER_LINEAR,
        )  # shape: (em_h, em_w)

        # Upsample the mask too (for alpha compositing)
        feather_mask_em = cv2.resize(
            feather_mask_nano,
            (em_w, em_h),
            interpolation=cv2.INTER_LINEAR,
        )
        # ── Contrast-adjust original EM for display ───────────────────────────
        em_display = auto_adjust_contrast_complete(em_img_original)

        # ── Normalize to [0, 1] ───────────────────────────────────────────────
        em_norm   = em_display.astype(np.float32) / 255.0
        nano_norm = nano_upsampled.astype(np.float32) / 255.0
        # nano_norm already has feathered edges; feather_mask_em drives alpha

        # ── Save raw outputs ───────────────────────────────────────────────────
        cv2.imwrite('warped_nano_nano_res.png',
                    nano_channel_feathered.astype(np.uint8))
        cv2.imwrite('warped_nano_em_res.png', nano_upsampled.astype(np.uint8))
        cv2.imwrite('em_image.png', em_display)

        # ── Save NanoSIMS and EM crop for registration (grayscale) ────────────
        cv2.imwrite('nanosims_for_registration.png', nano_img.astype(np.uint8))
        cv2.imwrite('em_crop_for_registration.png', em_img.astype(np.uint8))

        # ── 1. Red-Cyan composite ─────────────────────────────────────────────
        composite_rgb = np.stack([
            em_norm,
            np.clip(em_norm * 0.8 + nano_norm * 0.5, 0, 1),
            np.clip(em_norm * 0.8 + nano_norm * 0.8, 0, 1),
        ], axis=-1)
        cv2.imwrite('overlay_composite.png',
                    (composite_rgb * 255).astype(np.uint8)[:, :, ::-1])

        # ── Helper: smooth alpha overlay (replaces masked_where) ──────────────
        def alpha_overlay(ax_obj, base_norm, nano_n, mask, cmap='hot',
                          max_alpha=0.55, intensity_thresh=0.01):
            """
            Overlay nano on base using per-pixel alpha = mask * visibility,
            producing smooth edges with no sawtooth artefacts.
            """
            ax_obj.imshow(base_norm, cmap='gray', vmin=0, vmax=1)

            # RGBA nano layer
            cmap_obj  = plt.get_cmap(cmap)
            nano_rgba = cmap_obj(nano_n)                   # (H, W, 4)
            nano_rgba = nano_rgba.copy()

            # Combine feather mask with a soft intensity ramp
            # (avoids the hard 0.05 threshold → no binary edge)
            intensity_alpha = np.clip(nano_n / (intensity_thresh * 4), 0, 1)
            intensity_alpha = cv2.GaussianBlur(
                intensity_alpha.astype(np.float32), (0, 0), sigmaX=2)

            combined_alpha = mask * intensity_alpha * max_alpha
            nano_rgba[..., 3] = combined_alpha.astype(np.float32)

            return ax_obj.imshow(nano_rgba, vmin=0, vmax=1)

        # ── 2. Matplotlib panel ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 gridspec_kw={'wspace': 0.05})
        fig.patch.set_facecolor('black')

        axes[0].imshow(em_norm, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('EM Image', color='white', fontsize=13, pad=8)
        axes[0].axis('off')

        im_nano = axes[1].imshow(nano_norm, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('NanoSIMS (warped)', color='white', fontsize=13, pad=8)
        axes[1].axis('off')
        plt.colorbar(im_nano, ax=axes[1], fraction=0.046, pad=0.04,
                     label='Intensity').ax.yaxis.label.set_color('white')

        im_overlay = alpha_overlay(axes[2], em_norm, nano_norm, feather_mask_em)
        axes[2].set_title('EM + NanoSIMS Overlay', color='white', fontsize=13, pad=8)
        axes[2].axis('off')
        # cbar = plt.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.04,
        #                     label='NanoSIMS Intensity')
        # cbar.ax.yaxis.label.set_color('white')
        # cbar.ax.tick_params(colors='white')

        plt.savefig('final_overlay_panel.png',
                    bbox_inches='tight', dpi=200,
                    facecolor='black', edgecolor='none')
        plt.close()

        # ── 3. Single high-res overlay ────────────────────────────────────────
        fig2, ax = plt.subplots(figsize=(8, 8))
        fig2.patch.set_facecolor('black')
        im2 = alpha_overlay(ax, em_norm, nano_norm, feather_mask_em,
                            max_alpha=0.55)
        # cbar2 = plt.colorbar(im2, ax=ax, fraction=0.035, pad=0.02,
        #                      label='NanoSIMS Intensity')
        # cbar2.ax.yaxis.label.set_color('white')
        # cbar2.ax.tick_params(colors='white')
        ax.axis('off')
        plt.savefig('final_overlay_hires.png',
                    bbox_inches='tight', dpi=300,
                    facecolor='black', edgecolor='none')
        plt.close()

        print("✅ Saved: warped_nano_nano_res.png, warped_nano_em_res.png, "
              "em_image.png, overlay_composite.png, "
              "final_overlay_panel.png, final_overlay_hires.png, "
              "nanosims_for_registration.png, em_crop_for_registration.png")

    return warp_matrix

def demo2(args, em_img, nano_img):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    # path = 'Gary d83-DSPC/Gary1_3.nrrd'
    model = model.module
    model.to(DEVICE)
    model.eval()
    em_img_original = em_img.copy()
    with torch.no_grad():
        em_img = image_resize(em_img, width=nano_img.shape[1], height=nano_img.shape[0])
        em_img = auto_adjust_contrast_complete(em_img)
        nano_img = auto_adjust_contrast_complete(nano_img)
        wf_img = nano_img
        gt_img = em_img
        print(f"EM image shape: {em_img.shape}, Nano image shape: {nano_img.shape}")
        h, w = wf_img.shape[:2]

        image1 = load_image2(gt_img)
        image2 = load_image2(wf_img)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_up_fw = model(image1, image2)

        flow_up_bw = model(image2, image1)
        occlusion_mask = detect_occlusion(flow_up_fw, flow_up_bw)

        # Invert the mask so 1 = confident, 0 = bad
        confidence_mask = 1 - occlusion_mask
        print('confidence_map', confidence_mask.max(), confidence_mask.min(), confidence_mask.mean())
        warp_matrix = flow_to_affine_confidence(flow_up_fw, confidence_mask)
        image2_np = image2[0].permute(1, 2, 0).cpu().numpy()

        image_warped_nano = cv2.warpAffine(
            image2_np.astype(np.float32),
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )  # shape: (h, w, 3), values in [0, 255]

        # ── Upsample warped nano to original EM resolution ────────────────────
        em_h, em_w = em_img_original.shape[:2]
        nano_channel = image_warped_nano[:, :, 0]  # single channel, [0, 255]

        nano_upsampled = cv2.resize(
            nano_channel,
            (em_w, em_h),
            interpolation=cv2.INTER_LINEAR,

        )  # shape: (em_h, em_w), values in [0, 255]

        # ── Contrast-adjust original EM for display ───────────────────────────
        em_display = auto_adjust_contrast_complete(em_img_original)  # [0, 255] uint8

        # ── Normalize both to [0, 1] for matplotlib ───────────────────────────
        em_norm = em_display.astype(np.float32) / 255.0
        nano_norm = nano_upsampled.astype(np.float32) / 255.0

        # ── Save raw outputs ───────────────────────────────────────────────────
        cv2.imwrite('warped_nano_nano_res.png', nano_channel.astype(np.uint8))
        cv2.imwrite('warped_nano_em_res.png', nano_upsampled.astype(np.uint8))
        cv2.imwrite('em_image.png', em_display)

        # ── Save NanoSIMS and EM crop for registration (grayscale) ────────────
        cv2.imwrite('nanosims_for_registration.png', nano_img.astype(np.uint8))
        cv2.imwrite('em_crop_for_registration.png', em_img.astype(np.uint8))

        # ── 1. Red-Cyan composite (EM=gray, Nano=cyan) ────────────────────────
        composite_rgb = np.stack([
            em_norm,  # R
            np.clip(em_norm * 0.8 + nano_norm * 0.5, 0, 1),  # G (blend)
            np.clip(em_norm * 0.8 + nano_norm * 0.8, 0, 1),  # B (cyan tint)
        ], axis=-1)
        cv2.imwrite('overlay_composite.png',
                    (composite_rgb * 255).astype(np.uint8)[:, :, ::-1])

        # ── 2. Matplotlib: EM gray + NanoSIMS hot colormap ────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 gridspec_kw={'wspace': 0.05})
        fig.patch.set_facecolor('black')

        # Panel 1: EM only
        axes[0].imshow(em_norm, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('EM Image', color='white', fontsize=13, pad=8)
        axes[0].axis('off')

        # Panel 2: NanoSIMS only (hot colormap)
        im_nano = axes[1].imshow(nano_norm, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('NanoSIMS (warped)', color='white', fontsize=13, pad=8)
        axes[1].axis('off')
        plt.colorbar(im_nano, ax=axes[1], fraction=0.046, pad=0.04,
                     label='Intensity').ax.yaxis.label.set_color('white')

        # Panel 3: Overlay — EM as gray base, NanoSIMS as semi-transparent hot
        axes[2].imshow(em_norm, cmap='gray', vmin=0, vmax=1)
        # Mask near-zero nano pixels so background stays clean
        nano_masked = np.ma.masked_where(nano_norm < 0.05, nano_norm)
        im_overlay = axes[2].imshow(nano_masked, cmap='hot',
                                    vmin=0, vmax=1, alpha=0.55)
        axes[2].set_title('EM + NanoSIMS Overlay', color='white', fontsize=13, pad=8)
        axes[2].axis('off')
        cbar = plt.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.04,
                            label='NanoSIMS Intensity')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        plt.savefig('final_overlay_panel.png',
                    bbox_inches='tight', dpi=200,
                    facecolor='black', edgecolor='none')
        plt.close()

        # ── 3. Single high-res overlay (publication-style) ────────────────────
        fig2, ax = plt.subplots(figsize=(8, 8))
        fig2.patch.set_facecolor('black')
        ax.imshow(em_norm, cmap='gray', vmin=0, vmax=1)
        nano_masked = np.ma.masked_where(nano_norm < 0.05, nano_norm)
        im2 = ax.imshow(nano_masked, cmap='hot', vmin=0, vmax=1, alpha=0.55)
        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.035, pad=0.02,
                             label='NanoSIMS Intensity')
        cbar2.ax.yaxis.label.set_color('white')
        cbar2.ax.tick_params(colors='white')
        ax.axis('off')
        plt.savefig('final_overlay_hires.png',
                    bbox_inches='tight', dpi=300,
                    facecolor='black', edgecolor='none')
        plt.close()

        print("✅ Saved: warped_nano_nano_res.png, warped_nano_em_res.png, "
              "em_image.png, overlay_composite.png, "
              "final_overlay_panel.png, final_overlay_hires.png, "
              "nanosims_for_registration.png, em_crop_for_registration.png")
    return warp_matrix
def overlay_signal_on_em(nanosims_path, em_path, step1_result, warp_matrix,
                         signal_name='127I', em_res_nm=6., nano_res_nm=97.65625,
                         output_path=None, alpha=0.5, percentile_clip=(1, 99)):
    """
    Apply the estimated transformation to a specific NanoSIMS signal and overlay it
    on the EM image in GREEN at EM resolution.

    Args:
        nanosims_path: Path to NanoSIMS NRRD file
        em_path: Path to EM image
        step1_result: Result dictionary from find_coarse_alignment
        warp_matrix: 2x3 affine transformation matrix from Step 2
        signal_name: Name of the signal to overlay (e.g., '127I')
        em_res_nm: EM resolution in nm/pixel
        nano_res_nm: NanoSIMS resolution in nm/pixel
        output_path: Path to save the overlay image
        alpha: Transparency of the overlay (0-1)
        percentile_clip: Tuple for contrast adjustment (min_percentile, max_percentile)

    Returns:
        overlay_image: RGB image with signal overlaid in green
        signal_em_res: The transformed signal at EM resolution
    """
    print(f"\n{'=' * 60}")
    print(f"OVERLAYING {signal_name} SIGNAL ON EM IMAGE")
    print(f"{'=' * 60}\n")

    # 1. Load NanoSIMS data
    print(f"Loading NanoSIMS file: {nanosims_path}")
    nano_data, header = nrrd.read(nanosims_path)

    # Parse signal information
    mass_symbols = header.get('Mims_mass_symbols', '').split(' ')
    print(f"Available signals: {mass_symbols}")

    # Find the target signal index
    try:
        signal_idx = mass_symbols.index(signal_name)
        print(f"Found {signal_name} at index: {signal_idx}")
    except ValueError:
        raise ValueError(f"Signal '{signal_name}' not found in mass symbols: {mass_symbols}")

    # 2. Extract the specific signal
    original_shape = nano_data.shape
    if len(original_shape) == 3:
        signal_image = nano_data[:, :, signal_idx].astype(np.float32)
        is_3d = False
    elif len(original_shape) == 4:
        num_planes = original_shape[2]
        print(f"3D data detected with {num_planes} planes. Summing across planes.")
        signal_image = nano_data[:, :, :, signal_idx].astype(np.float32)
        signal_image = np.sum(signal_image, axis=2)
        is_3d = True
    else:
        raise ValueError(f"Unexpected data shape: {original_shape}")

    print(f"Signal shape: {signal_image.shape}")
    print(f"Signal range: [{signal_image.min():.2f}, {signal_image.max():.2f}]")

    # 3. Apply special preprocessing
    if 'MC3 spleen' in em_path:
        print("\n⚠️  Applying special preprocessing for MC3 spleen...")
        signal_image = cv2.flip(signal_image, 0)
        signal_image = signal_image[:-256, :-256]

    # 4. Apply Step 1 orientation transformation
    orientation_label = step1_result['orientation_label']
    if orientation_label and orientation_label != "Normal_0":
        print(f"\n📐 Applying Step 1 orientation: {orientation_label}")
        dtype = signal_image.dtype
        signal_image = apply_orientation_transform(signal_image.astype(np.float32), orientation_label).astype(dtype)
        print(f"After orientation: {signal_image.shape}")

    # 5. Crop to the refined region (256x256)
    signal_image = signal_image[:256, :256]
    print(f"Cropped to refined region: {signal_image.shape}")

    # 6. Apply Step 2 warp transformation
    print(f"\n🔧 Applying Step 2 warp transformation...")
    signal_image = cv2.GaussianBlur(signal_image, (0, 0), 1)
    height, width = signal_image.shape

    signal_warped = cv2.warpAffine(
        signal_image.astype(np.float32),
        warp_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)
    print(f"Warped signal range: [{signal_warped.min():.2f}, {signal_warped.max():.2f}]")

    # 7. Upscale to EM resolution
    print(f"\n🔍 Upscaling to EM resolution...")
    ratio = nano_res_nm / em_res_nm
    target_h_em = int(height * ratio)
    target_w_em = int(width * ratio)

    signal_em_res = cv2.resize(
        signal_warped,
        (target_w_em, target_h_em),
        interpolation=cv2.INTER_NEAREST
    )
    print(f"Upscaled to EM resolution: {signal_em_res.shape} (ratio: {ratio:.2f})")

    # 8. Load and prepare EM image
    print(f"\nLoading EM image: {em_path}")
    em_img = cv2.imread(em_path, 0)
    em_refined_patch = step1_result['em_refined_patch']

    if em_refined_patch.shape != signal_em_res.shape:
        print(f"⚠️  Resizing EM patch from {em_refined_patch.shape} to {signal_em_res.shape}")
        em_refined_patch = cv2.resize(em_refined_patch, (target_w_em, target_h_em))

    em_normalized = cv2.normalize(em_refined_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    em_rgb = cv2.cvtColor(em_normalized, cv2.COLOR_GRAY2RGB)

    # 9. Normalize signal for visualization
    print(f"\n🎨 Preparing overlay...")
    vmin = 0.
    vmax = 10.
    signal_clipped = np.clip(signal_em_res, vmin, vmax)
    signal_normalized = ((signal_clipped - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    print(f"Signal visualization range: [{vmin:.2f}, {vmax:.2f}]")

    # 10. Create "I Green" LUT:
    #     Low value  → transparent / white  (R=255, G=255, B=255, A=0)
    #     High value → pure green           (R=0,   G=255, B=0,   A=255)
    #
    #     Strategy:
    #       - RGB channel: interpolate white (1,1,1) → green (0,1,0)
    #         R: 1 → 0  (decreases with intensity)
    #         G: 1 → 1  (stays at 1 throughout)
    #         B: 1 → 0  (decreases with intensity)
    #       - Alpha channel: 0 → 1  (fully transparent at low, opaque at high)
    from matplotlib.colors import LinearSegmentedColormap

    i_green_colors = [
        (0.0, (0.80, 1.00, 0.80)),  # light green
        (0.4, (0.40, 1.00, 0.40)),  # mid green
        (0.7, (0.10, 1.00, 0.10)),  # vivid green
        (1.0, (0.00, 0.80, 0.00)),  # deep green
    ]
    i_green_cmap = LinearSegmentedColormap.from_list(
        'i_cyan',
        [(pos, col) for pos, col in i_green_colors]
    )

    signal_norm_01 = signal_normalized / 255.0  # float [0, 1]

    # RGB from colormap
    green_colored  = i_green_cmap(signal_norm_01)                        # RGBA float [0,1]
    green_overlay  = (green_colored[:, :, :3] * 255).astype(np.uint8)    # RGB  uint8
    green_overlay_bgr = cv2.cvtColor(green_overlay, cv2.COLOR_RGB2BGR)   # BGR  for OpenCV

    # Alpha mask: proportional to signal intensity (low → transparent, high → opaque)
    # This replaces the hard binary mask — no separate GaussianBlur needed on the mask.
    # signal_alpha = signal_norm_01.astype(np.float32)   # 0 = transparent, 1 = opaque
    gamma = 0.6
    signal_alpha = np.power(signal_norm_01, gamma).astype(np.float32)
    # Optional: apply a gentle blur only to the alpha mask for soft edges
    signal_alpha_soft = cv2.GaussianBlur(signal_alpha, (15, 15), 5)
    signal_alpha_soft = np.clip(signal_alpha_soft, 0.0, 1.0)

    # 11. Blend: where signal is low → EM shows through; where signal is high → green
    overlay_image = em_rgb.copy().astype(np.float32)
    for c in range(3):
        overlay_image[:, :, c] = (
            (1 - alpha * signal_alpha_soft) * overlay_image[:, :, c]
            + alpha * signal_alpha_soft * green_overlay_bgr[:, :, c]
        )

    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)

    # Keep a hard mask for contour drawing only
    signal_mask_hard = (signal_normalized > 0).astype(np.float32)

    # 12. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')

    title_kwargs = dict(fontsize=12, fontweight='bold', color='white')

    # Row 1
    axes[0, 0].imshow(em_normalized, cmap='gray')
    axes[0, 0].set_title('EM Image (High Res)', **title_kwargs)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(signal_em_res, cmap='Greens')
    axes[0, 1].set_title(f'{signal_name} Signal (Transformed)', **title_kwargs)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(signal_normalized, cmap='Greens')
    axes[0, 2].set_title(f'{signal_name} (Normalized)', **title_kwargs)
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Overlay: EM + {signal_name} (I Green LUT)', **title_kwargs)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(em_normalized, cmap='gray')
    axes[1, 1].contour(signal_mask_hard, levels=[0.5], colors='lime', linewidths=1)
    axes[1, 1].set_title('Signal Boundary on EM', **title_kwargs)
    axes[1, 1].axis('off')

    im = axes[1, 2].imshow(signal_em_res, cmap=i_green_cmap)
    axes[1, 2].set_title(f'{signal_name} Heatmap (I Green LUT)', **title_kwargs)
    axes[1, 2].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()

    # 13. Save results
    if output_path is None:
        output_path = f"overlay_{signal_name}_on_EM.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n✅ Overlay visualization saved to: {output_path}")

    overlay_only_path = output_path.replace('.png', '_overlay_only.png')
    cv2.imwrite(overlay_only_path, overlay_image)
    print(f"✅ Overlay image saved to: {overlay_only_path}")

    plt.show()

    return overlay_image, signal_em_res

def overlay_signal_on_em2(nanosims_path, em_path, step1_result, warp_matrix,
                         signal_name='127I', em_res_nm=6., nano_res_nm=97.65625,
                         output_path=None, alpha=0.5, percentile_clip=(1, 99)):
    """
    Apply the estimated transformation to a specific NanoSIMS signal and overlay it
    on the EM image in GREEN at EM resolution.

    Args:
        nanosims_path: Path to NanoSIMS NRRD file
        em_path: Path to EM image
        step1_result: Result dictionary from find_coarse_alignment
        warp_matrix: 2x3 affine transformation matrix from Step 2
        signal_name: Name of the signal to overlay (e.g., '127I')
        em_res_nm: EM resolution in nm/pixel
        nano_res_nm: NanoSIMS resolution in nm/pixel
        output_path: Path to save the overlay image
        alpha: Transparency of the overlay (0-1)
        percentile_clip: Tuple for contrast adjustment (min_percentile, max_percentile)

    Returns:
        overlay_image: RGB image with signal overlaid in green
        signal_em_res: The transformed signal at EM resolution
    """
    print(f"\n{'=' * 60}")
    print(f"OVERLAYING {signal_name} SIGNAL ON EM IMAGE")
    print(f"{'=' * 60}\n")

    # 1. Load NanoSIMS data
    print(f"Loading NanoSIMS file: {nanosims_path}")
    nano_data, header = nrrd.read(nanosims_path)

    # Parse signal information
    mass_symbols = header.get('Mims_mass_symbols', '').split(' ')
    print(f"Available signals: {mass_symbols}")

    # Find the target signal index
    try:
        signal_idx = mass_symbols.index(signal_name)
        print(f"Found {signal_name} at index: {signal_idx}")
    except ValueError:
        raise ValueError(f"Signal '{signal_name}' not found in mass symbols: {mass_symbols}")

    # 2. Extract the specific signal
    original_shape = nano_data.shape
    if len(original_shape) == 3:
        signal_image = nano_data[:, :, signal_idx].astype(np.float32)
        is_3d = False
    elif len(original_shape) == 4:
        num_planes = original_shape[2]
        print(f"3D data detected with {num_planes} planes. Summing across planes.")
        signal_image = nano_data[:, :, :, signal_idx].astype(np.float32)
        signal_image = np.sum(signal_image, axis=2)
        is_3d = True
    else:
        raise ValueError(f"Unexpected data shape: {original_shape}")

    print(f"Signal shape: {signal_image.shape}")
    print(f"Signal range: [{signal_image.min():.2f}, {signal_image.max():.2f}]")

    # 3. Apply special preprocessing
    if 'MC3 spleen' in em_path:
        print("\n⚠️  Applying special preprocessing for MC3 spleen...")
        signal_image = cv2.flip(signal_image, 0)
        signal_image = signal_image[:-256, :-256]

    # 4. Apply Step 1 orientation transformation
    orientation_label = step1_result['orientation_label']
    if orientation_label and orientation_label != "Normal_0":
        print(f"\n📐 Applying Step 1 orientation: {orientation_label}")
        dtype = signal_image.dtype
        signal_image = apply_orientation_transform(signal_image.astype(np.float32), orientation_label).astype(dtype)
        print(f"After orientation: {signal_image.shape}")

    # 5. Crop to the refined region (256x256)
    signal_image = signal_image[:256, :256]
    print(f"Cropped to refined region: {signal_image.shape}")

    # 6. Apply Step 2 warp transformation
    print(f"\n🔧 Applying Step 2 warp transformation...")
    signal_image = cv2.GaussianBlur(signal_image, (0, 0), 1)
    height, width = signal_image.shape

    signal_warped = cv2.warpAffine(
        signal_image.astype(np.float32),
        warp_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)
    print(f"Warped signal range: [{signal_warped.min():.2f}, {signal_warped.max():.2f}]")

    # 7. Upscale to EM resolution
    print(f"\n🔍 Upscaling to EM resolution...")
    ratio = nano_res_nm / em_res_nm
    target_h_em = int(height * ratio)
    target_w_em = int(width * ratio)

    signal_em_res = cv2.resize(
        signal_warped,
        (target_w_em, target_h_em),
        interpolation=cv2.INTER_NEAREST
    )
    print(f"Upscaled to EM resolution: {signal_em_res.shape} (ratio: {ratio:.2f})")

    # 8. Load and prepare EM image
    print(f"\nLoading EM image: {em_path}")
    em_img = cv2.imread(em_path, 0)
    em_refined_patch = step1_result['em_refined_patch']

    if em_refined_patch.shape != signal_em_res.shape:
        print(f"⚠️  Resizing EM patch from {em_refined_patch.shape} to {signal_em_res.shape}")
        em_refined_patch = cv2.resize(em_refined_patch, (target_w_em, target_h_em))

    em_normalized = cv2.normalize(em_refined_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    em_rgb = cv2.cvtColor(em_normalized, cv2.COLOR_GRAY2RGB)

    # 9. Normalize signal for visualization
    print(f"\n🎨 Preparing overlay...")
    vmin = 0.
    vmax = 10.
    signal_clipped = np.clip(signal_em_res, vmin, vmax)
    signal_normalized = ((signal_clipped - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    print(f"Signal visualization range: [{vmin:.2f}, {vmax:.2f}]")

    # 10. Create "I Green" LUT
    from matplotlib.colors import LinearSegmentedColormap

    i_green_colors = [
        (0.0, (0.80, 1.00, 0.80)),  # light green
        (0.4, (0.40, 1.00, 0.40)),  # mid green
        (0.7, (0.10, 1.00, 0.10)),  # vivid green
        (1.0, (0.00, 0.80, 0.00)),  # deep green
    ]
    i_green_cmap = LinearSegmentedColormap.from_list(
        'i_cyan',
        [(pos, col) for pos, col in i_green_colors]
    )

    signal_norm_01 = signal_normalized / 255.0  # float [0, 1]

    # RGB from colormap
    green_colored     = i_green_cmap(signal_norm_01)                        # RGBA float [0,1]
    green_overlay     = (green_colored[:, :, :3] * 255).astype(np.uint8)    # RGB  uint8
    green_overlay_bgr = cv2.cvtColor(green_overlay, cv2.COLOR_RGB2BGR)      # BGR  for OpenCV

    # Alpha mask: proportional to signal intensity (low → transparent, high → opaque)
    gamma = 0.6
    signal_alpha      = np.power(signal_norm_01, gamma).astype(np.float32)
    signal_alpha_soft = cv2.GaussianBlur(signal_alpha, (15, 15), 5)
    signal_alpha_soft = np.clip(signal_alpha_soft, 0.0, 1.0)

    # 11. Blend: where signal is low → EM shows through; where signal is high → green
    overlay_image = em_rgb.copy().astype(np.float32)
    for c in range(3):
        overlay_image[:, :, c] = (
            (1 - alpha * signal_alpha_soft) * overlay_image[:, :, c]
            + alpha * signal_alpha_soft * green_overlay_bgr[:, :, c]
        )

    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)

    # Keep a hard mask for contour drawing only
    signal_mask_hard = (signal_normalized > 0).astype(np.float32)

    # ── NEW: build the signal-only image using the SAME contrast pipeline ──────
    # Render signal_em_res with i_green_cmap at the same vmin/vmax used for the
    # overlay, then convert to a plain uint8 BGR image (no alpha compositing).
    signal_green_rgba  = i_green_cmap(signal_norm_01)                        # RGBA [0,1]
    signal_green_rgb   = (signal_green_rgba[:, :, :3] * 255).astype(np.uint8)
    signal_green_bgr   = cv2.cvtColor(signal_green_rgb, cv2.COLOR_RGB2BGR)
    # ──────────────────────────────────────────────────────────────────────────

    # 12. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')

    title_kwargs = dict(fontsize=12, fontweight='bold', color='white')

    # Row 1
    axes[0, 0].imshow(em_normalized, cmap='gray')
    axes[0, 0].set_title('EM Image (High Res)', **title_kwargs)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(signal_em_res, cmap='Greens')
    axes[0, 1].set_title(f'{signal_name} Signal (Transformed)', **title_kwargs)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(signal_normalized, cmap='Greens')
    axes[0, 2].set_title(f'{signal_name} (Normalized)', **title_kwargs)
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Overlay: EM + {signal_name} (I Green LUT)', **title_kwargs)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(em_normalized, cmap='gray')
    axes[1, 1].contour(signal_mask_hard, levels=[0.5], colors='lime', linewidths=1)
    axes[1, 1].set_title('Signal Boundary on EM', **title_kwargs)
    axes[1, 1].axis('off')

    im = axes[1, 2].imshow(signal_em_res, cmap=i_green_cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].set_title(f'{signal_name} Heatmap (I Green LUT)', **title_kwargs)
    axes[1, 2].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()

    # 13. Save results
    if output_path is None:
        output_path = f"overlay_{signal_name}_on_EM.png"

    base_path = output_path.replace('.png', '')

    # ── (a) Main diagnostic figure ────────────────────────────────────────────
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n✅ Overlay visualization saved to: {output_path}")

    # ── (b) Overlay-only image (EM + green signal) ────────────────────────────
    overlay_only_path = f"{base_path}_overlay_only.png"
    cv2.imwrite(overlay_only_path, overlay_image)
    print(f"✅ Overlay image saved to: {overlay_only_path}")

    # ── (c) Signal image — same contrast as overlay, green LUT ───────────────
    signal_image_path = f"{base_path}_signal_green.png"
    cv2.imwrite(signal_image_path, signal_green_bgr)
    print(f"✅ Signal image (green LUT, same contrast) saved to: {signal_image_path}")

    # ── (d) Colorbar — saved as a standalone figure ───────────────────────────
    colorbar_path = f"{base_path}_colorbar.png"
    fig_cb, ax_cb = plt.subplots(figsize=(1.2, 5))
    fig_cb.patch.set_facecolor('white')
    ax_cb.set_facecolor('white')

    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    norm_cb = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = mcm.ScalarMappable(cmap=i_green_cmap, norm=norm_cb)
    sm.set_array([])

    cbar_standalone = fig_cb.colorbar(sm, cax=ax_cb)
    cbar_standalone.set_ticks([])  # remove all ticks
    cbar_standalone.outline.set_edgecolor('black')  # clean border
    cbar_standalone.set_label(signal_name, color='black', fontsize=11)

    fig_cb.savefig(colorbar_path, dpi=300, bbox_inches='tight',
                   facecolor=fig_cb.get_facecolor())
    plt.close(fig_cb)
    print(f"✅ Colorbar saved to: {colorbar_path}")

    plt.show()

    return overlay_image, signal_em_res


def apply_remap_to_all_signals(nano_path, warp_matrix, orientation_label=None,
                               output_path=None, s32_index=None,
                               crop_coords=None, em_path=None):
    """
    Apply orientation transformation and non-rigid remap fields to all signals in the NRRD file.
    """
    print(f"\n{'=' * 60}")
    print("APPLYING NON-RIGID TRANSFORMATIONS TO ALL NANOSIMS SIGNALS")
    print(f"{'=' * 60}\n")

    # 1. Load NRRD file
    print(f"Loading NRRD file: {nano_path}")
    nano_data, header = nrrd.read(nano_path)
    nano_data = nano_data[:256, :256, ...]  # Crop to refined region for processing
    print(f"Original data shape: {nano_data.shape}")
    print(f"Data type: {nano_data.dtype}")

    # 2. Parse signal information
    mass_symbols = header.get('Mims_mass_symbols', '').split(' ')
    print(f"Available signals: {mass_symbols}")

    # 3. Determine data dimensions
    original_shape = nano_data.shape

    if len(original_shape) == 3:
        height, width, num_signals = original_shape
        num_planes = 1
        is_3d = False
    elif len(original_shape) == 4:
        height, width, num_planes, num_signals = original_shape
        is_3d = True
    else:
        raise ValueError(f"Unexpected data shape: {original_shape}")

    print(f"Image dimensions: {height}x{width}")
    print(f"Number of signals: {num_signals}")
    if is_3d:
        print(f"Number of planes: {num_planes}")

    # 4. Apply special preprocessing
    if em_path and 'MC3 spleen' in em_path:
        print("\n⚠️  Applying special preprocessing for MC3 spleen...")
        if not is_3d:
            nano_data = cv2.flip(nano_data, 0)
            nano_data = nano_data[:-256, :-256, :]
        else:
            for plane_idx in range(num_planes):
                nano_data[:, :, plane_idx, :] = cv2.flip(nano_data[:, :, plane_idx, :], 0)
            nano_data = nano_data[:-256, :-256, :, :]

        height, width = nano_data.shape[0], nano_data.shape[1]
        print(f"After preprocessing: {nano_data.shape}")

    # 5. Apply Step 1 orientation transformation
    if orientation_label and orientation_label != "Normal_0":
        print(f"\n📐 Applying Step 1 orientation: {orientation_label}")

        if not is_3d:
            oriented_data = np.zeros_like(nano_data)
            for signal_idx in range(num_signals):
                signal_image = nano_data[:, :, signal_idx]
                oriented_signal = apply_orientation_transform(signal_image.astype(np.float32), orientation_label)
                oriented_data[:, :, signal_idx] = oriented_signal.astype(signal_image.dtype)
            nano_data = oriented_data
        else:
            oriented_data = np.zeros_like(nano_data)
            for plane_idx in range(num_planes):
                for signal_idx in range(num_signals):
                    signal_image = nano_data[:, :, plane_idx, signal_idx]
                    oriented_signal = apply_orientation_transform(signal_image.astype(np.float32), orientation_label)
                    oriented_data[:, :, plane_idx, signal_idx] = oriented_signal.astype(signal_image.dtype)
            nano_data = oriented_data

        height, width = nano_data.shape[0], nano_data.shape[1]
        print(f"After orientation: {nano_data.shape}")
    else:
        print("\n📐 No orientation transformation needed (Normal_0)")

    # 6. Initialize output array
    warped_data = np.zeros_like(nano_data)

    # 7. Apply non-rigid remap transformation
    print(f"\n🔧 Applying non-rigid remap transformation...")

    if not is_3d:
        for signal_idx in range(num_signals):
            signal_name = mass_symbols[signal_idx] if signal_idx < len(mass_symbols) else f"Signal_{signal_idx}"
            print(f"  Processing {signal_name} ({signal_idx + 1}/{num_signals})...")
            signal_image = nano_data[:, :, signal_idx]
            warped_signal = cv2.warpAffine(
                signal_image.astype(np.float32),
                warp_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)
            warped_data[:, :, signal_idx] = warped_signal
    else:
        for plane_idx in range(num_planes):
            for signal_idx in range(num_signals):
                signal_name = mass_symbols[signal_idx] if signal_idx < len(mass_symbols) else f"Signal_{signal_idx}"
                print(
                    f"  Processing {signal_name} - Plane {plane_idx + 1}/{num_planes} ({signal_idx + 1}/{num_signals})...")

                signal_image = nano_data[:, :, plane_idx, signal_idx]
                warped_signal = cv2.warpAffine(
                    signal_image.astype(np.float32),
                    warp_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
                warped_data[:, :, plane_idx, signal_idx] = warped_signal

    warped_data = warped_data.astype(nano_data.dtype)

    print(f"\n✅ All transformations completed!")
    print(f"Final output shape: {warped_data.shape}")

    # 9. Save to new NRRD file
    if output_path is None:
        input_path = Path(nano_path)
        output_path = input_path.parent / f"{input_path.stem}_aligned.nrrd"

    print(f"\nSaving aligned data to: {output_path}")

    header_copy = header.copy()
    header_copy['alignment_method'] = 'TPS+OpticalFlow (Non-Rigid)'
    if orientation_label:
        header_copy['orientation_transform'] = orientation_label
    if crop_coords:
        header_copy['crop_coords'] = crop_coords
    header_copy['aligned_from'] = str(nano_path)
    if em_path:
        header_copy['em_reference'] = str(em_path)

    nrrd.write(str(output_path), warped_data, header_copy)
    print(f"✅ Saved successfully!")

    return warped_data, header_copy


def apply_orientation_transform(image, orientation_label):
    """
    Apply orientation transformation based on the label from Step 1.

    Args:
        image: Input image (2D array)
        orientation_label: String like "Normal_0", "Flipped_90", etc.

    Returns:
        Transformed image
    """
    parts = orientation_label.split('_')
    is_flipped = (parts[0] == "Flipped")
    angle = int(parts[1])

    # Apply flip first if needed
    if is_flipped:
        image = cv2.flip(image, 1)  # Horizontal flip

    # Apply rotation
    if angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # angle == 0: no rotation needed

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='../models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--category', help="save warped images")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--occlusion', action='store_true', help='predict occlusion masks')
    
    # Add new arguments for em_path, nano_path, em_res, nano_res
    parser.add_argument('--em_path', type=str, required=True, help='Path to EM image file')
    parser.add_argument('--nano_path', type=str, required=True, help='Path to NanoSIMS NRRD file')
    parser.add_argument('--em_res', type=float, required=True, help='EM resolution in nm per pixel')
    parser.add_argument('--nano_res', type=float, required=True, help='NanoSIMS resolution in nm per pixel')
    parser.add_argument('--channel', type=str, required=True, help='Name of the NanoSIMS signal to align (e.g., "32S")')

    args = parser.parse_args()
    
    # Use command-line arguments for paths and resolutions
    em_path = args.em_path
    nano_path = args.nano_path
    em_res = args.em_res
    nano_res = args.nano_res

    from step1_v2 import find_coarse_alignment

    # 1. Run Step 1
    step1_data = find_coarse_alignment(em_path, nano_path, nano_res_nm=nano_res, em_res_nm=em_res, signal='32S')
    em_refined = step1_data['em_refined_patch']
    nanosims_refined = step1_data['corrected_nanosims']
    warp_matrix = demo(args, em_refined, nanosims_refined)
    # warp_matrix = flow_to_affine(flow_up)
    apply_remap_to_all_signals(
        nano_path, warp_matrix,
        orientation_label=step1_data['orientation_label'],
        output_path=nano_path.replace('.nrrd', '_aligned.nrrd')
    )