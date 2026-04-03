import nrrd
import numpy as np
import cv2
def auto_adjust_contrast(image, auto_threshold=5000):
    """
    Auto adjust contrast for a 2D grayscale numpy image.

    This function implements the same algorithm as ImageJ's auto contrast adjustment.

    Parameters:
    -----------
    image : numpy.ndarray
        2D grayscale image (numpy array)
    auto_threshold : int, optional
        Threshold parameter for auto adjustment (default: 5000, same as ImageJ)

    Returns:
    --------
    tuple: (min_value, max_value)
        The calculated minimum and maximum values for contrast adjustment
    """

    # Ensure image is 2D
    if len(image.shape) != 2:
        raise ValueError("Input image must be 2D grayscale")

    # Convert to appropriate data type for histogram calculation
    if image.dtype == np.float32 or image.dtype == np.float64:
        # For float images, scale to 0-255 range for histogram
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_scaled = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image_scaled = np.zeros_like(image, dtype=np.uint8)
        bins = 256
        hist_range = (0, 255)
    else:
        # For integer images
        image_scaled = image.astype(np.uint8) if image.max() <= 255 else (image * 255 / image.max()).astype(np.uint8)
        bins = 256
        hist_range = (0, 255)

    # Calculate histogram
    histogram, bin_edges = np.histogram(image_scaled.flatten(), bins=bins, range=hist_range)

    # Get image statistics
    pixel_count = image.size
    hist_min = bin_edges[0]
    bin_size = bin_edges[1] - bin_edges[0]

    # Set parameters
    limit = pixel_count // 10
    threshold = pixel_count // auto_threshold

    # Find minimum intensity (hmin)
    hmin = 0
    for i in range(256):
        count = histogram[i]
        if count > limit:
            count = 0
        if count > threshold:
            hmin = i
            break

    # Find maximum intensity (hmax)
    hmax = 255
    for i in range(255, -1, -1):
        count = histogram[i]
        if count > limit:
            count = 0
        if count > threshold:
            hmax = i
            break

    # Calculate actual min/max values
    if hmax >= hmin:
        min_val = hist_min + hmin * bin_size
        max_val = hist_min + hmax * bin_size

        # If min equals max, use actual image min/max
        if min_val == max_val:
            min_val = float(image.min())
            max_val = float(image.max())

        # For float images, scale back to original range
        if image.dtype == np.float32 or image.dtype == np.float64:
            if img_max > img_min:
                min_val = img_min + (min_val / 255.0) * (img_max - img_min)
                max_val = img_min + (max_val / 255.0) * (img_max - img_min)
            else:
                min_val = img_min
                max_val = img_max
    else:
        # Fallback to image min/max
        min_val = float(image.min())
        max_val = float(image.max())

    return min_val, max_val


def apply_contrast_adjustment(image, min_val, max_val):
    """
    Apply contrast adjustment to image using calculated min/max values.

    Parameters:
    -----------
    image : numpy.ndarray
        2D grayscale image
    min_val : float
        Minimum value for contrast adjustment
    max_val : float
        Maximum value for contrast adjustment

    Returns:
    --------
    numpy.ndarray: Contrast-adjusted image
    """
    if max_val <= min_val:
        return image
    image = image.astype(np.float32)
    # Clip values to min/max range and normalize to 0-1
    adjusted = np.clip(image, min_val, max_val)
    adjusted = (adjusted - min_val) / (max_val - min_val)

    # Scale back to original data type range
    # if image.dtype == np.uint8:
    #     return (adjusted * 255).astype(np.uint8)
    # elif image.dtype == np.uint16:
    #     return (adjusted * 65535).astype(np.uint16)
    # else:
    #     return adjusted.astype(image.dtype)
    return (adjusted * 255).astype(np.uint8)


def auto_adjust_contrast_complete(image, auto_threshold=5000, min_val=None, max_val=None, return_min_max=False):
    """
    Complete auto contrast adjustment function that both calculates and applies the adjustment.

    Parameters:
    -----------
    image : numpy.ndarray
        2D grayscale image
    auto_threshold : int, optional
        Threshold parameter for auto adjustment (default: 5000)

    Returns:
    --------
    numpy.ndarray: Contrast-adjusted image
    """
    if min_val is None:
        min_val, max_val = auto_adjust_contrast(image, auto_threshold)
    if return_min_max:
        return apply_contrast_adjustment(image, min_val, max_val), min_val, max_val
    return apply_contrast_adjustment(image, min_val, max_val)
def read_nanosims_file(file_path):
    """
    Reads a Nanosims .nrrd file and returns the data and header information.

    Parameters:
    file_path (str): The path to the .nrrd file.

    Returns:
    tuple: A tuple containing the data array and header dictionary.
    """
    try:
        data, header = nrrd.read(file_path)
        return data, header
    except Exception as e:
        print(f"Error reading Nanosims file {file_path}: {str(e)}")
        return None, None

def min_max_normalize(data):
    """
    Normalizes the data array to the range [0, 1] using min-max normalization.

    Parameters:
    data (numpy.ndarray): The input data array.

    Returns:
    numpy.ndarray: The normalized data array.
    """
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min == 0:
        return data  # Avoid division by zero
    normalized_data = (data - data_min) / (data_max - data_min)
    normalized_data = normalized_data * 255  # Scale to [0, 255]
    return normalized_data.astype('uint8')