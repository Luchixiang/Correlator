package com.haibolab.nansimsemcorrelation;

import com.nrims.data.Mims_Reader;
import com.nrims.data.Nrrd_Reader;
import com.nrims.data.Nrrd_Writer;
import com.nrims.data.Opener;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.io.File;
import java.io.IOException;

/**
 * Applies the Step-1 orientation transform + Step-2 affine warp matrix to
 * every mass channel in a NanoSIMS file and saves the result as a new .nrrd.
 *
 * Mirrors Python apply_remap_to_all_signals() in stabilize_nanosims_affine.py.
 *
 * Pipeline per signal (per plane for 4-D data):
 *   1. Crop to top-left 256×256   (nano_data = nano_data[:256, :256, ...])
 *   2. Apply orientation transform  (same label from Step 1 coarse alignment)
 *   3. Apply affine warp            (WARP_INVERSE_MAP, BORDER_CONSTANT=0)
 *   4. Write all channels to <stem>_aligned.nrrd via Nrrd_Writer
 */
public class AllSignalsWarper {

    private final String nanoPath;
    private final double[][] warpMatrix;   // 2×3 affine from FineAlignmentResult
    private final String orientationLabel; // e.g. "Flipped_90" from CoarseAlignmentResult

    public AllSignalsWarper(String nanoPath, double[][] warpMatrix, String orientationLabel) {
        this.nanoPath        = nanoPath;
        this.warpMatrix      = warpMatrix;
        this.orientationLabel = orientationLabel;
    }

    /**
     * Runs the full warp pipeline and saves to outputPath.
     * If outputPath is null it defaults to <stem>_aligned.nrrd next to the input.
     *
     * @return the saved output File, or null on error
     */
    public File apply(String outputPath) {


        // ── 1. Open reader ────────────────────────────────────────────────────
        File inputFile = new File(nanoPath);
        boolean isNrrd = nanoPath.toLowerCase().endsWith(".nrrd");
        Opener reader;
        try {
            reader = isNrrd ? new Nrrd_Reader(inputFile) : new Mims_Reader(inputFile);
        } catch (IOException e) {
            IJ.error("AllSignalsWarper", "Cannot open NanoSIMS file:\n" + e.getMessage());
            return null;
        }

        int width     = reader.getWidth();
        int height    = reader.getHeight();
        int nMasses   = reader.getNMasses();
        int nPlanes   = reader.getNImages();   // planes (z / time frames)
        String[] massSymbols = reader.getMassSymbols();
        String[] massNames   = reader.getMassNames();

        IJ.log("File: " + nanoPath);
        IJ.log(String.format("Dimensions: %d × %d,  masses: %d,  planes: %d",
                width, height, nMasses, nPlanes));

        // ── 2. Crop region: top-left 256×256  (matches Python [:256, :256, ...]) ──
        int cropW = Math.min(256, width);
        int cropH = Math.min(256, height);

        // ── 3. Build one ImagePlus per mass channel (stack = planes) ─────────
        ImagePlus[] outputChannels = new ImagePlus[nMasses];

        for (int massIdx = 0; massIdx < nMasses; massIdx++) {
            String sigName = (massSymbols != null && massIdx < massSymbols.length)
                    ? massSymbols[massIdx]
                    : (massNames != null && massIdx < massNames.length ? massNames[massIdx]
                       : "Signal_" + massIdx);

            IJ.showProgress(massIdx, nMasses);
            IJ.showStatus("Warping " + sigName + " (" + (massIdx+1) + "/" + nMasses + ")…");

            ImageStack warpedStack = new ImageStack(cropW, cropH);

            for (int planeIdx = 0; planeIdx < nPlanes; planeIdx++) {
                IJ.log(String.format("  %s  plane %d/%d", sigName, planeIdx+1, nPlanes));

                // Read raw plane
                Object pixels;
                try {
                    reader.setStackIndex(planeIdx);
                    pixels = reader.getPixels(massIdx);
                } catch (Exception e) {
                    IJ.log("  ⚠ read error — skipping plane " + planeIdx + ": " + e.getMessage());
                    warpedStack.addSlice(new FloatProcessor(cropW, cropH));
                    continue;
                }

                FloatProcessor fp = toFloat(pixels, width, height);

                // ── a. Crop to 256×256 ──────────────────────────────────────


                // ── b. Orientation transform (Step 1) ───────────────────────
                FloatProcessor oriented = applyOrientation(fp, orientationLabel);
                oriented.setRoi(0, 0, cropW, cropH);
                FloatProcessor oriented_cropped = (FloatProcessor) oriented.crop();

                // ── c. Affine warp (Step 2, WARP_INVERSE_MAP, BORDER_CONSTANT=0) ─
                FloatProcessor warped = warpAffineInverse(oriented_cropped, warpMatrix,
                        oriented_cropped.getWidth(), oriented_cropped.getHeight());

                warpedStack.addSlice("plane_" + (planeIdx + 1), warped);
            }

            ImagePlus channel = new ImagePlus(sigName, warpedStack);
            outputChannels[massIdx] = channel;
        }

        reader.close();
        IJ.showProgress(1.0);

        // ── 4. Write output NRRD ──────────────────────────────────────────────
        if (outputPath == null) {
            String stem = inputFile.getName().replaceAll("\\.[^.]+$", "");
            outputPath  = inputFile.getParent() + File.separator + stem + "_aligned.nrrd";
        }

        File outFile = new File(outputPath);
        try {
            // Nrrd_Writer(Opener) constructor re-uses the original header metadata
            Opener readerForMeta;
            readerForMeta = isNrrd ? new Nrrd_Reader(inputFile) : new Mims_Reader(inputFile);
            Nrrd_Writer writer = new Nrrd_Writer(readerForMeta);
            writer.save(outputChannels,
                        outFile.getParent() + File.separator,
                        outFile.getName());
            readerForMeta.close();
            IJ.log("\n✅ Saved aligned NanoSIMS file: " + outputPath);
        } catch (Exception e) {
            IJ.error("AllSignalsWarper", "Failed to save NRRD:\n" + e.getMessage());
            e.printStackTrace();
            return null;
        }

        return outFile;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Orientation transform  (mirrors Python apply_orientation_transform)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Applies the Step-1 orientation (flip + rotation) to a single plane.
     * Labels: "Normal_0" | "Normal_90" | "Normal_180" | "Normal_270"
     *        "Flipped_0" | "Flipped_90" | "Flipped_180" | "Flipped_270"
     */
    static FloatProcessor applyOrientation(FloatProcessor fp, String label) {
        if (label == null || label.equals("Normal_0")) return fp;

        String[] parts = label.split("_");
        boolean flipped = parts[0].equals("Flipped");
        int angle = Integer.parseInt(parts[1]);

        FloatProcessor out = (FloatProcessor) fp.duplicate();

        // 1. Horizontal flip first (matches Python cv2.flip(img, 1))
        if (flipped) out.flipHorizontal();

        // 2. Rotation (ImageJ rotate() is CCW; Python uses CW codes)
        //    Python ROTATE_90_CLOCKWISE  → ImageJ rotate(-90)
        //    Python ROTATE_180           → ImageJ rotate(180)
        //    Python ROTATE_90_CCW        → ImageJ rotate(90)
        if (angle == 90)       out.rotate(-90);
        else if (angle == 180) out.rotate(180);
        else if (angle == 270) out.rotate(90);

        return out;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pure-Java warpAffine with WARP_INVERSE_MAP + bilinear + BORDER_CONSTANT=0
    // Mirrors Python cv2.warpAffine(..., flags=INTER_LINEAR+WARP_INVERSE_MAP,
    //                                    borderMode=BORDER_CONSTANT, borderValue=0)
    // ─────────────────────────────────────────────────────────────────────────

    static FloatProcessor warpAffineInverse(FloatProcessor src, double[][] M,
                                            int outW, int outH) {
        float[] outPx = new float[outW * outH];
        int srcW = src.getWidth(), srcH = src.getHeight();
        float[] srcPx = (float[]) src.getPixelsCopy();

        double m00 = M[0][0], m01 = M[0][1], m02 = M[0][2];
        double m10 = M[1][0], m11 = M[1][1], m12 = M[1][2];

        for (int dy = 0; dy < outH; dy++) {
            for (int dx = 0; dx < outW; dx++) {
                // Inverse map: dst → src
                double sx = m00*dx + m01*dy + m02;
                double sy = m10*dx + m11*dy + m12;

                // BORDER_CONSTANT = 0: out-of-bounds pixels become 0
                if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) {
                    outPx[dy * outW + dx] = 0f;
                    continue;
                }

                // Bilinear interpolation
                int x0 = (int) sx, y0 = (int) sy;
                int x1 = Math.min(x0 + 1, srcW - 1);
                int y1 = Math.min(y0 + 1, srcH - 1);
                double fx = sx - x0, fy = sy - y0;

                float v = (float)(
                        (1-fx)*(1-fy) * srcPx[y0*srcW + x0]
                      + fx   *(1-fy) * srcPx[y0*srcW + x1]
                      + (1-fx)* fy   * srcPx[y1*srcW + x0]
                      + fx   * fy    * srcPx[y1*srcW + x1]);

                outPx[dy * outW + dx] = v;
            }
        }

        FloatProcessor result = new FloatProcessor(outW, outH, outPx, null);
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Convert whatever pixel array the reader returns into a FloatProcessor. */
    private static FloatProcessor toFloat(Object pixels, int w, int h) {
        if (pixels instanceof float[]) {
            return new FloatProcessor(w, h, (float[]) pixels, null);
        } else if (pixels instanceof short[]) {
            short[] sp = (short[]) pixels;
            float[] fp = new float[sp.length];
            for (int i = 0; i < sp.length; i++) fp[i] = sp[i] & 0xFFFF;
            return new FloatProcessor(w, h, fp, null);
        } else if (pixels instanceof int[]) {
            int[] ip = (int[]) pixels;
            float[] fp = new float[ip.length];
            for (int i = 0; i < ip.length; i++) fp[i] = ip[i];
            return new FloatProcessor(w, h, fp, null);
        } else if (pixels instanceof byte[]) {
            byte[] bp = (byte[]) pixels;
            float[] fp = new float[bp.length];
            for (int i = 0; i < bp.length; i++) fp[i] = bp[i] & 0xFF;
            return new FloatProcessor(w, h, fp, null);
        } else {
            IJ.log("  ⚠ Unknown pixel type: " + pixels.getClass().getName() + " — returning zeros");
            return new FloatProcessor(w, h);
        }
    }
}

