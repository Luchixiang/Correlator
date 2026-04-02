package com.haibolab.nansimsemcorrelation;




import ij.process.ImageProcessor;
import ij.process.FloatProcessor;
import ij.process.ByteProcessor;
import ij.process.ShortProcessor;

public class ImageNormalizer {

    /**
     * Normalize image to 0-255 range using min-max normalization
     *
     * @param ip Input ImageProcessor
     * @return Normalized ImageProcessor as ByteProcessor (8-bit, 0-255)
     */
    public static ByteProcessor normalizeToByteRange(ImageProcessor ip) {
        return normalizeToByteRange(ip, null, null);
    }

    /**
     * Normalize image to 0-255 range using specified min-max values
     *
     * @param ip Input ImageProcessor
     * @param minVal Minimum value for normalization (null = use image min)
     * @param maxVal Maximum value for normalization (null = use image max)
     * @return Normalized ImageProcessor as ByteProcessor (8-bit, 0-255)
     */
    public static ByteProcessor normalizeToByteRange(ImageProcessor ip, Double minVal, Double maxVal) {
        int width = ip.getWidth();
        int height = ip.getHeight();

        // Find actual min and max if not provided
        if (minVal == null || maxVal == null) {
            double[] minMax = findMinMax(ip);
            if (minVal == null) minVal = minMax[0];
            if (maxVal == null) maxVal = minMax[1];
        }

        // Create output byte processor
        ByteProcessor result = new ByteProcessor(width, height);

        // Handle edge case where min equals max
        if (maxVal.equals(minVal)) {
            // All pixels get the same value (middle of range)
            byte fillValue = (byte) 127;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    result.set(x, y, fillValue & 0xFF);
                }
            }
            return result;
        }

        // Perform normalization
        double range = maxVal - minVal;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = ip.getPixelValue(x, y);

                // Clip to min-max range
                if (value < minVal) value = minVal;
                if (value > maxVal) value = maxVal;

                // Normalize to 0-255
                double normalized = ((value - minVal) / range) * 255.0;
                int byteValue = (int) Math.round(normalized);

                // Ensure value is in valid range
                if (byteValue < 0) byteValue = 0;
                if (byteValue > 255) byteValue = 255;

                result.set(x, y, byteValue);
            }
        }

        return result;
    }

    /**
     * Normalize image to 0-1 range (float)
     *
     * @param ip Input ImageProcessor
     * @return Normalized ImageProcessor as FloatProcessor (32-bit float, 0.0-1.0)
     */
    public static FloatProcessor normalizeToFloatRange(ImageProcessor ip) {
        return normalizeToFloatRange(ip, null, null);
    }

    /**
     * Normalize image to 0-1 range using specified min-max values
     *
     * @param ip Input ImageProcessor
     * @param minVal Minimum value for normalization (null = use image min)
     * @param maxVal Maximum value for normalization (null = use image max)
     * @return Normalized ImageProcessor as FloatProcessor (32-bit float, 0.0-1.0)
     */
    public static FloatProcessor normalizeToFloatRange(ImageProcessor ip, Double minVal, Double maxVal) {
        int width = ip.getWidth();
        int height = ip.getHeight();

        // Find actual min and max if not provided
        if (minVal == null || maxVal == null) {
            double[] minMax = findMinMax(ip);
            if (minVal == null) minVal = minMax[0];
            if (maxVal == null) maxVal = minMax[1];
        }

        // Create output float processor
        FloatProcessor result = new FloatProcessor(width, height);

        // Handle edge case where min equals max
        if (maxVal.equals(minVal)) {
            // All pixels get the same value (middle of range)
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    result.setf(x, y, 0.5f);
                }
            }
            return result;
        }

        // Perform normalization
        double range = maxVal - minVal;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = ip.getPixelValue(x, y);

                // Clip to min-max range
                if (value < minVal) value = minVal;
                if (value > maxVal) value = maxVal;

                // Normalize to 0-1
                float normalized = (float) ((value - minVal) / range);

                result.setf(x, y, normalized);
            }
        }

        return result;
    }

    /**
     * Normalize image to custom range
     *
     * @param ip Input ImageProcessor
     * @param targetMin Target minimum value
     * @param targetMax Target maximum value
     * @return Normalized ImageProcessor as FloatProcessor
     */
    public static FloatProcessor normalizeToCustomRange(ImageProcessor ip, double targetMin, double targetMax) {
        return normalizeToCustomRange(ip, null, null, targetMin, targetMax);
    }

    /**
     * Normalize image to custom range using specified source min-max values
     *
     * @param ip Input ImageProcessor
     * @param sourceMin Source minimum value (null = use image min)
     * @param sourceMax Source maximum value (null = use image max)
     * @param targetMin Target minimum value
     * @param targetMax Target maximum value
     * @return Normalized ImageProcessor as FloatProcessor
     */
    public static FloatProcessor normalizeToCustomRange(ImageProcessor ip, Double sourceMin, Double sourceMax,
                                                        double targetMin, double targetMax) {
        int width = ip.getWidth();
        int height = ip.getHeight();

        // Find actual min and max if not provided
        if (sourceMin == null || sourceMax == null) {
            double[] minMax = findMinMax(ip);
            if (sourceMin == null) sourceMin = minMax[0];
            if (sourceMax == null) sourceMax = minMax[1];
        }

        // Create output float processor
        FloatProcessor result = new FloatProcessor(width, height);

        // Handle edge case where min equals max
        if (sourceMax.equals(sourceMin)) {
            float fillValue = (float) ((targetMin + targetMax) / 2.0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    result.setf(x, y, fillValue);
                }
            }
            return result;
        }

        // Perform normalization
        double sourceRange = sourceMax - sourceMin;
        double targetRange = targetMax - targetMin;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = ip.getPixelValue(x, y);

                // Clip to source min-max range
                if (value < sourceMin) value = sourceMin;
                if (value > sourceMax) value = sourceMax;

                // Normalize to target range
                double normalized = ((value - sourceMin) / sourceRange) * targetRange + targetMin;

                result.setf(x, y, (float) normalized);
            }
        }

        return result;
    }

    /**
     * Normalize image in-place (modifies the original ImageProcessor)
     *
     * @param ip ImageProcessor to normalize (will be modified)
     * @param minVal Minimum value for normalization (null = use image min)
     * @param maxVal Maximum value for normalization (null = use image max)
     * @param targetMin Target minimum value (default 0)
     * @param targetMax Target maximum value (default 255 for byte, 65535 for short, 1.0 for float)
     */
    public static void normalizeInPlace(ImageProcessor ip, Double minVal, Double maxVal,
                                        Double targetMin, Double targetMax) {
        int width = ip.getWidth();
        int height = ip.getHeight();

        // Find actual min and max if not provided
        if (minVal == null || maxVal == null) {
            double[] minMax = findMinMax(ip);
            if (minVal == null) minVal = minMax[0];
            if (maxVal == null) maxVal = minMax[1];
        }

        // Set default target range based on image type
        if (targetMin == null) targetMin = 0.0;
        if (targetMax == null) {
            if (ip instanceof ByteProcessor) {
                targetMax = 255.0;
            } else if (ip instanceof ShortProcessor) {
                targetMax = 65535.0;
            } else if (ip instanceof FloatProcessor) {
                targetMax = 1.0;
            } else {
                targetMax = 255.0;
            }
        }

        // Handle edge case where min equals max
        if (maxVal.equals(minVal)) {
            double fillValue = (targetMin + targetMax) / 2.0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    ip.putPixelValue(x, y, fillValue);
                }
            }
            return;
        }

        // Perform normalization
        double sourceRange = maxVal - minVal;
        double targetRange = targetMax - targetMin;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = ip.getPixelValue(x, y);

                // Clip to source min-max range
                if (value < minVal) value = minVal;
                if (value > maxVal) value = maxVal;

                // Normalize to target range
                double normalized = ((value - minVal) / sourceRange) * targetRange + targetMin;

                ip.putPixelValue(x, y, normalized);
            }
        }
    }

    /**
     * Find minimum and maximum pixel values in an image
     *
     * @param ip Input ImageProcessor
     * @return Array containing [min, max]
     */
    public static double[] findMinMax(ImageProcessor ip) {
        int width = ip.getWidth();
        int height = ip.getHeight();

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = ip.getPixelValue(x, y);
                if (value < min) min = value;
                if (value > max) max = value;
            }
        }

        return new double[]{min, max};
    }

}
