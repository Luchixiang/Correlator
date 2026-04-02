package com.haibolab.nansimsemcorrelation;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

public class CoarseAligner {

    private final double emResNm;
    private final double nanoResNm;

    public CoarseAligner(double emResNm, double nanoResNm) {
        this.emResNm = emResNm;
        this.nanoResNm = nanoResNm;
    }

    public CoarseAlignmentResult align(ImagePlus emImage, ImagePlus nanoImage) {
        IJ.log("=== Starting Coarse Alignment ===");

        // 1. Rescale EM DOWN to match NanoSIMS resolution
        //    scale_factor = em_res_nm / nano_res_nm  (e.g. 6/97.65 ≈ 0.061)
        double scaleFactor = emResNm / nanoResNm;
        ImagePlus emScaled = scaleImage(emImage, scaleFactor);
        IJ.log("EM scaled by " + scaleFactor + " -> " + emScaled.getWidth() + "x" + emScaled.getHeight());

        // 2. Preprocess images:
        //    EM: normalize + auto-contrast (no invert)
        //    Nano: normalize + auto-contrast + invert (255 - nano_norm, matching Python)
        ImageProcessor emProc = preprocessImage(emScaled.getProcessor(), false);
        ImageProcessor nanoProc = preprocessImage(nanoImage.getProcessor(), true);

        // 3. Try all 8 orientations of NanoSIMS, find best match
        OrientationResult bestMatch = findBestOrientation(emProc, nanoProc);
        IJ.log("Best match score: " + bestMatch.score
                + ", Orientation: " + bestMatch.label);

        // 4. Calculate coarse coordinates back in original EM resolution
        int x = (int)(bestMatch.x / scaleFactor);
        int y = (int)(bestMatch.y / scaleFactor);
        int w = (int)(nanoProc.getWidth() / scaleFactor);
        int h = (int)(nanoProc.getHeight() / scaleFactor);

        IJ.log("Coarse coords (original res): x=" + x + " y=" + y + " w=" + w + " h=" + h);

        // 5. Refine the top-left 256x256 region of NanoSIMS
        RefinementResult refined = refineSmallRegion(
                emImage, bestMatch.nanoAligned, x, y, w, h
        );

        return new CoarseAlignmentResult(
                emImage,
                refined.emPatch,
                refined.nanoPatch,
                x, y, w, h,
                bestMatch.label
        );
    }

    private ImagePlus scaleImage(ImagePlus imp, double scale) {
        int newW = (int)(imp.getWidth() * scale);
        int newH = (int)(imp.getHeight() * scale);
        // Clamp to at least 1x1
        newW = Math.max(1, newW);
        newH = Math.max(1, newH);
        ImageProcessor ip = imp.getProcessor().resize(newW, newH);
        return new ImagePlus("scaled", ip);
    }

    /**
     * Preprocess image: min-max normalize to [0,255] then auto-contrast.
     * If invert==true, inverts the result (255 - value), matching Python's
     * "nano_norm = 255 - nano_norm".
     */
    private ImageProcessor preprocessImage(ImageProcessor ip, boolean invert) {
        ImageProcessor result = ip.duplicate().convertToByteProcessor();

        // 1. Min-max normalize to [0,255]
        double min = result.getMin();
        double max = result.getMax();
        if (max > min) {
            result.setMinAndMax(min, max);
            result = result.convertToByteProcessor(true);
        }

        // 2. Auto-contrast (matching Python auto_adjust_contrast_complete)
        ContrastEnhancer.autoAdjustContrast(result);

        // 3. Invert for NanoSIMS (Python: nano_norm = 255 - nano_norm)
        if (invert) {
            result.invert();
        }

        return result;
    }

    private OrientationResult findBestOrientation(
            ImageProcessor emProc, ImageProcessor nanoProc) {

        OrientationResult best = new OrientationResult();
        best.score = -1;

        // Generate 8 orientations: [normal, flipped] x [0, 90, 180, 270]
        ImageProcessor[][] variations = generateOrientations(nanoProc);
        String[] labels = {
                "Normal_0", "Normal_90", "Normal_180", "Normal_270",
                "Flipped_0", "Flipped_90", "Flipped_180", "Flipped_270"
        };

        int labelIdx = 0;
        for (int i = 0; i < variations.length; i++) {
            for (int j = 0; j < variations[i].length; j++) {
                ImageProcessor nano = variations[i][j];

                // Skip if template is larger than image (matching Python)
                if (emProc.getHeight() < nano.getHeight() ||
                        emProc.getWidth() < nano.getWidth()) {
                    labelIdx++;
                    continue;
                }

                TemplateMatchResult match = templateMatch(emProc, nano);

                if (match.score > best.score) {
                    best.score = match.score;
                    best.x = match.x;
                    best.y = match.y;
                    best.label = labels[labelIdx];
                    best.nanoAligned = nano;
                }
                labelIdx++;
            }
        }

        return best;
    }

    private ImageProcessor[][] generateOrientations(ImageProcessor nano) {
        ImageProcessor[][] result = new ImageProcessor[2][4];

        // Normal orientations
        result[0][0] = nano.duplicate();
        result[0][1] = rotateClockwise(nano);
        result[0][2] = rotate180(nano);
        result[0][3] = rotateCounterClockwise(nano);

        // Flipped orientations (horizontal flip first, then rotate)
        ImageProcessor flipped = nano.duplicate();
        flipped.flipHorizontal();
        result[1][0] = flipped.duplicate();
        result[1][1] = rotateClockwise(flipped);
        result[1][2] = rotate180(flipped);
        result[1][3] = rotateCounterClockwise(flipped);

        return result;
    }

    private ImageProcessor rotateClockwise(ImageProcessor ip) {
        ImageProcessor result = ip.duplicate();
        result.rotate(-90); // ImageJ rotate(-90) = clockwise 90
        return result;
    }

    private ImageProcessor rotate180(ImageProcessor ip) {
        ImageProcessor result = ip.duplicate();
        result.rotate(180);
        return result;
    }

    private ImageProcessor rotateCounterClockwise(ImageProcessor ip) {
        ImageProcessor result = ip.duplicate();
        result.rotate(90); // ImageJ rotate(90) = counter-clockwise 90
        return result;
    }

    private TemplateMatchResult templateMatch(ImageProcessor image, ImageProcessor template) {
        return OpenCVTemplateMatching.match(image, template);
    }

    /**
     * Refines alignment for the top-left 256x256 patch of the NanoSIMS image.
     * Scales EM down to match NanoSIMS resolution for the search
     * (matches Python refine_small_region).
     */
    private RefinementResult refineSmallRegion(
            ImagePlus emHighRes, ImageProcessor nanoAligned,
            int x, int y, int w, int h) {

        IJ.log("\n--- Refinement Step: Aligning Top-Left (256x256) ---");

        // 1. Extract 256x256 patch from nano (top-left)
        ImageProcessor nanoPatch = cropProcessor(nanoAligned, 0, 0,
                Math.min(256, nanoAligned.getWidth()),
                Math.min(256, nanoAligned.getHeight()));

        // 2. Calculate scale ratio (nano_res / em_res)
        //    e.g. 97.65 / 6 ≈ 16.27  — how many EM pixels per 1 nano pixel
        double ratio = nanoResNm / emResNm;

        // 3. Define search region in high-res EM with generous margins
        int marginNanoPx = 128;
        int marginEmPx = (int)(marginNanoPx * ratio);
        int patchSpanEm = (int)(256 * ratio);

        int searchX = Math.max(0, x - marginEmPx);
        int searchY = Math.max(0, y - marginEmPx);
        int searchW = patchSpanEm + 2 * marginEmPx;
        int searchH = patchSpanEm + 2 * marginEmPx;

        // Clamp to image bounds
        searchW = Math.min(searchW, emHighRes.getWidth() - searchX);
        searchH = Math.min(searchH, emHighRes.getHeight() - searchY);

        if (searchW <= 0 || searchH <= 0) {
            IJ.log("Warning: Search region out of bounds — using coarse patch directly.");
            ImageProcessor emPatch = cropProcessor(emHighRes.getProcessor(), x, y,
                    Math.min((int)(256 * ratio), emHighRes.getWidth() - x),
                    Math.min((int)(256 * ratio), emHighRes.getHeight() - y));
            return new RefinementResult(emPatch, nanoPatch);
        }

        ImageProcessor emSearch = cropProcessor(
                emHighRes.getProcessor(), searchX, searchY, searchW, searchH);

        // 4. Scale EM crop DOWN to NanoSIMS resolution
        int targetW = (int)(emSearch.getWidth() / ratio);
        int targetH = (int)(emSearch.getHeight() / ratio);
        targetW = Math.max(1, targetW);
        targetH = Math.max(1, targetH);

        // Normalize and resize (matching Python)
        ImageProcessor emSearchScaled = emSearch.duplicate().convertToByteProcessor();
        double smin = emSearchScaled.getMin(), smax = emSearchScaled.getMax();
        if (smax > smin) {
            emSearchScaled.setMinAndMax(smin, smax);
            emSearchScaled = emSearchScaled.convertToByteProcessor(true);
        }
        emSearchScaled = emSearchScaled.resize(targetW, targetH);
        ContrastEnhancer.autoAdjustContrast(emSearchScaled);

        // 5. Template match: template = nanoPatch (256x256), image = emSearchScaled
        if (emSearchScaled.getHeight() < nanoPatch.getHeight() ||
                emSearchScaled.getWidth() < nanoPatch.getWidth()) {
            IJ.log("Warning: Search region smaller than template after scaling. Using coarse crop.");
            ImageProcessor emPatch = cropProcessor(emHighRes.getProcessor(), x, y,
                    Math.min((int)(256 * ratio), emHighRes.getWidth() - x),
                    Math.min((int)(256 * ratio), emHighRes.getHeight() - y));
            return new RefinementResult(emPatch, nanoPatch);
        }

        TemplateMatchResult match = templateMatch(emSearchScaled, nanoPatch);
        IJ.log("Refinement Score: " + match.score);

        // 6. Map coordinates back to high-res EM
        int offsetXEm = (int)(match.x * ratio);
        int offsetYEm = (int)(match.y * ratio);

        int finalX = searchX + offsetXEm;
        int finalY = searchY + offsetYEm;
        int finalW = (int)(256 * ratio);
        int finalH = (int)(256 * ratio);

        // Clamp to image bounds
        finalW = Math.min(finalW, emHighRes.getWidth() - finalX);
        finalH = Math.min(finalH, emHighRes.getHeight() - finalY);
        finalX = Math.max(0, finalX);
        finalY = Math.max(0, finalY);

        IJ.log("Final EM Patch Coords: x=" + finalX + " y=" + finalY
                + " w=" + finalW + " h=" + finalH);

        ImageProcessor emPatch = cropProcessor(
                emHighRes.getProcessor(), finalX, finalY, finalW, finalH);

        return new RefinementResult(emPatch, nanoPatch);
    }

    private ImageProcessor cropProcessor(ImageProcessor ip, int x, int y, int w, int h) {
        ip.setRoi(x, y, w, h);
        return ip.crop();
    }

    static class OrientationResult {
        double score;
        int x, y;
        String label;
        ImageProcessor nanoAligned;
    }

    static class RefinementResult {
        ImageProcessor emPatch;
        ImageProcessor nanoPatch;

        RefinementResult(ImageProcessor em, ImageProcessor nano) {
            this.emPatch = em;
            this.nanoPatch = nano;
        }
    }
}
