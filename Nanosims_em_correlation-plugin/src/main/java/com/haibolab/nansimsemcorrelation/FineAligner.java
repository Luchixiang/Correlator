package com.haibolab.nansimsemcorrelation;

import ai.djl.ModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ij.IJ;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

import java.io.IOException;

import java.io.IOException;

public class FineAligner {

    /**
     * No configuration needed — the RAFT model is loaded from the bundled
     * "raft_modelv3.2.zip" resource inside the jar.
     */
    public FineAligner() { }

    /**
     * Fine alignment using RAFT optical flow.
     * Mirrors the Python demo() pipeline:
     *   1. Resize EM to NanoSIMS size
     *   2. Auto-contrast both
     *   3. RAFT forward + backward flow
     *   4. Occlusion detection → confidence mask
     *   5. flow_to_affine_confidence (RANSAC)
     *   6. warpAffine (WARP_INVERSE_MAP)
     */
    public FineAlignmentResult align(CoarseAlignmentResult coarseResult) {
        IJ.log("=== Starting Fine Alignment (RAFT) ===");

        ImageProcessor emPatch  = coarseResult.getEMRefinedPatch();
        ImageProcessor nanoPatch = coarseResult.getNanoAligned();

        int targetW = nanoPatch.getWidth();
        int targetH = nanoPatch.getHeight();

        // 1. Resize EM patch to match NanoSIMS size
        ImageProcessor emResized = emPatch.resize(targetW, targetH);

        // 2. Auto-contrast both images (matching Python auto_adjust_contrast_complete)
        ImageProcessor emContrast   = emResized.duplicate().convertToByteProcessor();
        ImageProcessor nanoContrast = nanoPatch.duplicate().convertToByteProcessor();
        ContrastEnhancer.autoAdjustContrast(emContrast);
        ContrastEnhancer.autoAdjustContrast(nanoContrast);

        IJ.log("EM shape: " + targetW + "x" + targetH
                + "  Nano shape: " + nanoContrast.getWidth() + "x" + nanoContrast.getHeight());

        // 3. Run RAFT optical flow
        float[] warpMatrixFlat;
        try {
            warpMatrixFlat = runRaft(emContrast, nanoContrast, targetW, targetH);
        } catch (Exception e) {
            IJ.error("RAFT optical flow failed: " + e.getMessage());
            e.printStackTrace();
            // Fallback: identity
            warpMatrixFlat = new float[]{ 1, 0, 0,  0, 1, 0 };
        }

        // 4. Convert flat [6] warp matrix to double[2][3]
        double[][] warpMatrix = new double[][]{
                { warpMatrixFlat[0], warpMatrixFlat[1], warpMatrixFlat[2] },
                { warpMatrixFlat[3], warpMatrixFlat[4], warpMatrixFlat[5] }
        };

        IJ.log(String.format("Warp matrix: [%.4f %.4f %.4f | %.4f %.4f %.4f]",
                warpMatrix[0][0], warpMatrix[0][1], warpMatrix[0][2],
                warpMatrix[1][0], warpMatrix[1][1], warpMatrix[1][2]));

        // 5. Apply warpAffine to the NanoSIMS patch (matching Python WARP_INVERSE_MAP)
        ImageProcessor warpedNano = applyWarpAffine(nanoContrast, warpMatrix, targetW, targetH);

        // 6. Create overlay for display
        ImageProcessor overlay = createOverlay(emContrast, warpedNano);

        return new FineAlignmentResult(
                warpedNano,
                emContrast,   // "emBoundaries" slot reused as preprocessed EM
                nanoContrast, // "nanoBoundaries" slot reused as preprocessed nano
                overlay,
                warpMatrix,
                "RAFT"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RAFT pipeline (mirrors Python demo())
    // ─────────────────────────────────────────────────────────────────────────

    private float[] runRaft(ImageProcessor em, ImageProcessor nano,
                            int w, int h) throws IOException, ModelException, TranslateException {

        try (NDManager manager = NDManager.newBaseManager()) {

            // Load the RAFT model from the bundled resource
            RaftFlowEstimator estimator = new RaftFlowEstimator();

            // Convert ImageProcessors to float[] NDArrays [H*W] grayscale
            float[] emPixels   = toFloatArray(em);
            float[] nanoPixels = toFloatArray(nano);

            NDArray emNd   = manager.create(emPixels,   new ai.djl.ndarray.types.Shape(h, w));
            NDArray nanoNd = manager.create(nanoPixels, new ai.djl.ndarray.types.Shape(h, w));

            // Forward flow: EM → Nano
            float[] flowFwFlat = estimator.estimateFlow(emNd, nanoNd);
            // Backward flow: Nano → EM
            float[] flowBwFlat = estimator.estimateFlow(nanoNd, emNd);

            // Occlusion detection → confidence mask (1 = confident, 0 = occluded)
            float[] confidenceMask = computeConfidenceMask(flowFwFlat, flowBwFlat, w, h);

            // flow_to_affine_confidence: RANSAC affine from dense flow
            return flowToAffineConfidence(flowFwFlat, confidenceMask, w, h);
        }
    }

    /**
     * Converts an ImageProcessor to a float[] normalized to [0, 255].
     */
    private float[] toFloatArray(ImageProcessor ip) {
        int n = ip.getWidth() * ip.getHeight();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = ip.getf(i);
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Occlusion detection (mirrors Python detect_occlusion)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Simple forward-backward consistency check.
     * confidence[i] = 1 if |fw + warp(fw, bw)| is small, else 0.
     * Mirrors Python detect_occlusion → returns confidence mask (1 - occlusion).
     */
    private float[] computeConfidenceMask(float[] flowFw, float[] flowBw, int w, int h) {
        // flowFw, flowBw: [2*H*W] = [u0,u1,...,v0,v1,...]  OR [H*W*2]?
        // The RAFT translator outputs [2, H, W] flattened → first H*W = u (x), next H*W = v (y)
        int n = w * h;
        float[] occlusion = new float[n];

        for (int i = 0; i < n; i++) {
            float fwU = flowFw[i];
            float fwV = flowFw[n + i];
            float bwU = flowBw[i];
            float bwV = flowBw[n + i];

            // Approximate: warp fw flow using bw flow (simplified — just sum)
            float sumU = fwU + bwU;
            float sumV = fwV + bwV;

            float fbMag  = sumU * sumU + sumV * sumV;
            float fwMag  = fwU * fwU  + fwV * fwV;
            float bwMag  = bwU * bwU  + bwV * bwV;

            // Occlusion criterion (matching Python mask1)
            boolean occ = fbMag > 0.01f * (fwMag + bwMag) + 0.5f;
            occlusion[i] = occ ? 1f : 0f;
        }

        // confidence = 1 - occlusion
        float[] confidence = new float[n];
        for (int i = 0; i < n; i++) {
            confidence[i] = 1f - occlusion[i];
        }
        return confidence;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // flow_to_affine_confidence (mirrors Python function)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Converts dense RAFT flow to a 2x3 affine matrix using OpenCV RANSAC,
     * filtered by the confidence mask.
     * Returns a float[6] = [m00,m01,m02, m10,m11,m12].
     */
    private float[] flowToAffineConfidence(float[] flow, float[] confidence, int w, int h) {
        int n = w * h;
        int subsample = 5;

        // Count valid (confident) points
        int validCount = 0;
        for (int i = 0; i < n; i++) {
            if (confidence[i] > 0) validCount++;
        }
        IJ.log("Confident points: " + validCount + " / " + n);

        if (validCount < 10) {
            IJ.log("Warning: Too few confident points. Using all points.");
            validCount = n;
            confidence = null; // use all
        }

        // Build subsampled src/dst point arrays
        java.util.List<float[]> srcList = new java.util.ArrayList<>();
        java.util.List<float[]> dstList = new java.util.ArrayList<>();

        int idx = 0;
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int i = py * w + px;
                if (confidence == null || confidence[i] > 0) {
                    if (idx % subsample == 0) {
                        float u = flow[i];       // x displacement
                        float v = flow[n + i];   // y displacement
                        srcList.add(new float[]{px, py});
                        dstList.add(new float[]{px + u, py + v});
                    }
                    idx++;
                }
            }
        }

        if (srcList.size() < 4) {
            IJ.log("Warning: Not enough points for RANSAC. Returning identity.");
            return new float[]{1, 0, 0, 0, 1, 0};
        }

        // Convert to OpenCV MatOfPoint2f
        MatOfPoint2f srcMat = new MatOfPoint2f();
        MatOfPoint2f dstMat = new MatOfPoint2f();
        Point[] srcPts = new Point[srcList.size()];
        Point[] dstPts = new Point[dstList.size()];
        for (int i = 0; i < srcList.size(); i++) {
            srcPts[i] = new Point(srcList.get(i)[0], srcList.get(i)[1]);
            dstPts[i] = new Point(dstList.get(i)[0], dstList.get(i)[1]);
        }
        srcMat.fromArray(srcPts);
        dstMat.fromArray(dstPts);

        // RANSAC affine estimation (matches Python cv2.estimateAffine2D)
        Mat inliersMat = new Mat();
        Mat affine = Calib3d.estimateAffine2D(srcMat, dstMat, inliersMat,
                Calib3d.RANSAC, 3.0, 2000, 0.99, 10);

        if (affine == null || affine.empty()) {
            IJ.log("Warning: Affine estimation failed. Returning identity.");
            return new float[]{1, 0, 0, 0, 1, 0};
        }

        // Log inlier ratio
        int inlierCount = Core.countNonZero(inliersMat);
        IJ.log(String.format("Affine RANSAC inlier ratio: %.2f%%",
                100.0 * inlierCount / srcList.size()));

        // Extract 2x3 matrix values and return
        return new float[]{
                (float) affine.get(0, 0)[0], (float) affine.get(0, 1)[0], (float) affine.get(0, 2)[0],
                (float) affine.get(1, 0)[0], (float) affine.get(1, 1)[0], (float) affine.get(1, 2)[0]
        };
    }

    // ─────────────────────────────────────────────────────────────────────────
    // warpAffine with WARP_INVERSE_MAP (mirrors Python cv2.warpAffine)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Applies a 2x3 affine transformation to the image using OpenCV,
     * with WARP_INVERSE_MAP flag (matches Python).
     */
    private ImageProcessor applyWarpAffine(ImageProcessor ip, double[][] matrix,
                                           int w, int h) {
        // Convert to OpenCV Mat
        Mat src = imageProcessorToMat(ip);
        Mat warpMat = new Mat(2, 3, CvType.CV_64FC1);
        warpMat.put(0, 0, matrix[0][0], matrix[0][1], matrix[0][2],
                matrix[1][0], matrix[1][1], matrix[1][2]);

        Mat dst = new Mat();
        // INTER_LINEAR | WARP_INVERSE_MAP  (flag value = 1 | 16 = 17)
        Imgproc.warpAffine(src, dst, warpMat, new Size(w, h),
                Imgproc.INTER_LINEAR | Imgproc.WARP_INVERSE_MAP,
                Core.BORDER_REPLICATE);

        return matToImageProcessor(dst, w, h);
    }

    private Mat imageProcessorToMat(ImageProcessor ip) {
        int w = ip.getWidth(), h = ip.getHeight();
        Mat mat = new Mat(h, w, CvType.CV_8UC1);
        byte[] pixels = new byte[w * h];
        for (int i = 0; i < w * h; i++) {
            pixels[i] = (byte)(ip.get(i % w, i / w) & 0xFF);
        }
        mat.put(0, 0, pixels);
        return mat;
    }

    private ImageProcessor matToImageProcessor(Mat mat, int w, int h) {
        byte[] pixels = new byte[w * h];
        mat.get(0, 0, pixels);
        ByteProcessor bp = new ByteProcessor(w, h);
        for (int i = 0; i < w * h; i++) {
            bp.set(i % w, i / w, pixels[i] & 0xFF);
        }
        return bp;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Overlay helper
    // ─────────────────────────────────────────────────────────────────────────

    private ImageProcessor createOverlay(ImageProcessor em, ImageProcessor nano) {
        int w = em.getWidth();
        int h = em.getHeight();
        ColorProcessor overlay = new ColorProcessor(w, h);

        final float MAX_ALPHA        = 0.55f;
        final float INTENSITY_THRESH = 0.04f; // intensity_thresh * 4  (matches Python)

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float emV   = em.getf(x, y)   / 255f;  // [0,1] gray
                float nanoV = nano.getf(x, y) / 255f;  // [0,1] NanoSIMS intensity

                // ── Gray EM base ─────────────────────────────────────────────
                float baseR = emV, baseG = emV, baseB = emV;

                // ── Hot colormap: 0→black, 0.33→red, 0.67→yellow, 1→white ───
                float hr, hg, hb;
                if (nanoV < 1f / 3f) {
                    float t = nanoV / (1f / 3f);
                    hr = t;  hg = 0f; hb = 0f;
                } else if (nanoV < 2f / 3f) {
                    float t = (nanoV - 1f / 3f) / (1f / 3f);
                    hr = 1f; hg = t;  hb = 0f;
                } else {
                    float t = (nanoV - 2f / 3f) / (1f / 3f);
                    hr = 1f; hg = 1f; hb = t;
                }

                // ── Intensity-driven alpha (no feather mask) ─────────────────
                float alpha = Math.min(1f, nanoV / INTENSITY_THRESH) * MAX_ALPHA;

                // ── Alpha-composite hot layer over gray base ──────────────────
                int r = Math.min(255, (int)((1f - alpha) * baseR * 255 + alpha * hr * 255));
                int g = Math.min(255, (int)((1f - alpha) * baseG * 255 + alpha * hg * 255));
                int b = Math.min(255, (int)((1f - alpha) * baseB * 255 + alpha * hb * 255));

                overlay.putPixel(x, y, new int[]{r, g, b});
            }
        }

        return overlay;
    }
}
