package com.haibolab.nansimsemcorrelation;

import ij.process.ImageProcessor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class OpenCVTemplateMatching {

    static {
        // Load OpenCV native library
        try {
            nu.pattern.OpenCV.loadLocally();
        } catch (Exception e) {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }
    }

    /**
     * Fast template matching using OpenCV
     * @param image Source image
     * @param template Template to find
     * @return Match result with score and position
     */
    public static TemplateMatchResult match(ImageProcessor image, ImageProcessor template) {
        // Convert ImageJ ImageProcessor to OpenCV Mat
        Mat imgMat = imageProcessorToMat(image);
        Mat tmpMat = imageProcessorToMat(template);

        // Check dimensions
        if (imgMat.cols() < tmpMat.cols() || imgMat.rows() < tmpMat.rows()) {
            return new TemplateMatchResult(-1, 0, 0);
        }

        // Create result matrix
        int resultCols = imgMat.cols() - tmpMat.cols() + 1;
        int resultRows = imgMat.rows() - tmpMat.rows() + 1;
        Mat result = new Mat(resultRows, resultCols, CvType.CV_32FC1);

        // Perform template matching
        // TM_CCOEFF_NORMED is equivalent to normalized cross-correlation
        Imgproc.matchTemplate(imgMat, tmpMat, result, Imgproc.TM_CCOEFF_NORMED);

        // Find best match location
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        // Clean up
        imgMat.release();
        tmpMat.release();
        result.release();

        return new TemplateMatchResult(
                mmr.maxVal,
                (int) mmr.maxLoc.x,
                (int) mmr.maxLoc.y
        );
    }

    /**
     * Multi-scale template matching for even better performance
     */
    public static TemplateMatchResult matchMultiScale(ImageProcessor image, ImageProcessor template) {
        int levels = 3;
        double bestScore = -1;
        int bestX = 0, bestY = 0;
        int searchRadius = 0;

        for (int level = levels - 1; level >= 0; level--) {
            double scale = Math.pow(0.5, level);

            // Scale images
            int imgW = (int)(image.getWidth() * scale);
            int imgH = (int)(image.getHeight() * scale);
            ImageProcessor imgScaled = image.duplicate().resize(imgW, imgH);

            int tmpW = (int)(template.getWidth() * scale);
            int tmpH = (int)(template.getHeight() * scale);
            ImageProcessor tmpScaled = template.duplicate().resize(tmpW, tmpH);

            TemplateMatchResult result;

            if (level == levels - 1) {
                // Full search at coarsest level
                result = match(imgScaled, tmpScaled);
            } else {
                // Refine search around previous best position
                bestX = bestX * 2;
                bestY = bestY * 2;
                searchRadius = 8;

                result = matchInRegion(imgScaled, tmpScaled, bestX, bestY, searchRadius);
            }

            bestScore = result.score;
            bestX = result.x;
            bestY = result.y;
        }

        // Final refinement at full resolution
        TemplateMatchResult finalResult = matchInRegion(image, template, bestX, bestY, 4);
        return finalResult;
    }

    /**
     * Match template in a specific region (for pyramid refinement)
     */
    private static TemplateMatchResult matchInRegion(ImageProcessor image, ImageProcessor template,
                                                     int centerX, int centerY, int radius) {
        int tmpW = template.getWidth();
        int tmpH = template.getHeight();

        // Calculate ROI bounds
        int roiX = Math.max(0, centerX - radius);
        int roiY = Math.max(0, centerY - radius);
        int roiW = Math.min(image.getWidth() - roiX, tmpW + 2 * radius);
        int roiH = Math.min(image.getHeight() - roiY, tmpH + 2 * radius);

        // Extract ROI
        image.setRoi(roiX, roiY, roiW, roiH);
        ImageProcessor roi = image.crop();

        // Match in ROI
        TemplateMatchResult result = match(roi, template);

        // Adjust coordinates to full image
        result.x += roiX;
        result.y += roiY;

        return result;
    }

    /**
     * Convert ImageJ ImageProcessor to OpenCV Mat
     */
    private static Mat imageProcessorToMat(ImageProcessor ip) {
        int width = ip.getWidth();
        int height = ip.getHeight();

        // Convert to byte array
        byte[] pixels;
        if (ip.getBitDepth() == 8) {
            pixels = (byte[]) ip.getPixels();
        } else {
            // Convert to 8-bit if necessary
            ImageProcessor ip8 = ip.convertToByteProcessor();
            pixels = (byte[]) ip8.getPixels();
        }

        // Create Mat and copy data
        Mat mat = new Mat(height, width, CvType.CV_8UC1);
        mat.put(0, 0, pixels);

        return mat;
    }

    /**
     * Alternative method using different matching methods
     */
    public static TemplateMatchResult matchWithMethod(ImageProcessor image, ImageProcessor template, int method) {
        Mat imgMat = imageProcessorToMat(image);
        Mat tmpMat = imageProcessorToMat(template);

        if (imgMat.cols() < tmpMat.cols() || imgMat.rows() < tmpMat.rows()) {
            return new TemplateMatchResult(-1, 0, 0);
        }

        int resultCols = imgMat.cols() - tmpMat.cols() + 1;
        int resultRows = imgMat.rows() - tmpMat.rows() + 1;
        Mat result = new Mat(resultRows, resultCols, CvType.CV_32FC1);

        Imgproc.matchTemplate(imgMat, tmpMat, result, method);

        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        // For some methods, minimum is the best match
        boolean useMin = (method == Imgproc.TM_SQDIFF || method == Imgproc.TM_SQDIFF_NORMED);
        double score = useMin ? -mmr.minVal : mmr.maxVal;
        Point loc = useMin ? mmr.minLoc : mmr.maxLoc;

        imgMat.release();
        tmpMat.release();
        result.release();

        return new TemplateMatchResult(score, (int) loc.x, (int) loc.y);
    }
}
