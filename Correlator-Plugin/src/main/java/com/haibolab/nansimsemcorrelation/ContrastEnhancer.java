package com.haibolab.nansimsemcorrelation;

import ij.process.ImageProcessor;

public class ContrastEnhancer {

    public static void autoAdjustContrast(ImageProcessor ip) {
        autoAdjustContrast(ip, 5000);
    }

    public static void autoAdjustContrast(ImageProcessor ip, int autoThreshold) {
        int[] histogram = ip.getHistogram();
        int pixelCount = ip.getPixelCount();
        int limit = pixelCount / 10;
        int threshold = pixelCount / autoThreshold;

        int hmin = 0;
        for (int i = 0; i < 256; i++) {
            int count = histogram[i];
            if (count > limit) {
                histogram[i] = 0;
            }
            if (count > threshold) {
                hmin = i;
                break;
            }
        }

        int hmax = 255;
        for (int i = 255; i >= 0; i--) {
            int count = histogram[i];
            if (count > limit) {
                histogram[i] = 0;
            }
            if (count > threshold) {
                hmax = i;
                break;
            }
        }

        // Apply contrast adjustment to actual pixel values
        if (hmax > hmin) {
            applyContrastAdjustment(ip, hmin, hmax);
        }
    }

    /**
     * Apply contrast adjustment to the image pixels.
     * This method modifies the actual pixel values, not just the display range.
     *
     * @param ip ImageProcessor to adjust
     * @param minVal Minimum value for contrast adjustment
     * @param maxVal Maximum value for contrast adjustment
     */
    private static void applyContrastAdjustment(ImageProcessor ip, double minVal, double maxVal) {
        if (maxVal <= minVal) {
            return;
        }

        int width = ip.getWidth();
        int height = ip.getHeight();
        double scale = 255.0 / (maxVal - minVal);

        // Process each pixel
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double pixelValue = ip.getPixelValue(x, y);

                // Clip to min/max range
                if (pixelValue < minVal) {
                    pixelValue = minVal;
                } else if (pixelValue > maxVal) {
                    pixelValue = maxVal;
                }

                // Normalize and scale to 0-255
                double adjusted = (pixelValue - minVal) * scale;

                // Set the new pixel value
                ip.putPixelValue(x, y, adjusted);
            }
        }

        // Update the display range to match the new pixel values
        ip.setMinAndMax(0, 255);
    }

    /**
     * Public method to apply contrast adjustment with custom min/max values.
     * Useful if you want to calculate min/max separately and apply them later.
     *
     * @param ip ImageProcessor to adjust
     * @param minVal Minimum value for contrast adjustment
     * @param maxVal Maximum value for contrast adjustment
     */
    public static void applyContrast(ImageProcessor ip, double minVal, double maxVal) {
        applyContrastAdjustment(ip, minVal, maxVal);
    }
}