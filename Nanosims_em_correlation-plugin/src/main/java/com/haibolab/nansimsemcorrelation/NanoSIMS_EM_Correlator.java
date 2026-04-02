package com.haibolab.nansimsemcorrelation;

import ij.*;
import ij.gui.*;
import ij.io.FileSaver;
import ij.io.OpenDialog;
import ij.plugin.PlugIn;
import ij.process.*;

import java.awt.*;
import java.io.File;

/**
 * NanoSIMS–EM Correlator FIJI plugin.
 *
 * Pipeline (mirrors Python step1_v2.py + stabilize_nanosims_affine.py):
 *   Step 1 – Coarse alignment via template matching (8 orientations, refinement)
 *   Step 2 – Fine alignment via RAFT optical flow + RANSAC affine
 *   Step 3 – Apply warp to all NanoSIMS signals and save aligned .nrrd
 */

public class NanoSIMS_EM_Correlator implements PlugIn {

    private static final String PLUGIN_NAME = "NanoSIMS-EM Correlator";

    // Persistent parameters (survive across plugin invocations)
    private static String lastEMPath        = "";
    private static String lastNanoPath      = "";
    private static double emResNm           = 6.0;
    private static double nanoResNm         = 97.65625;   // 25 µm / 256 px
    private static int    lastSignalIndex   = -1;         // -1 = auto-detect S32
    private static String lastSignalLabel   = "";

    @Override
    public void run(String arg) {
        // ── Dialog 1: file paths + resolution ───────────────────────────────
        if (!showPathDialog()) return;

        // ── Dialog 2: choose registration signal from the file ───────────────
        NanoSIMSReader reader = new NanoSIMSReader(lastNanoPath);
        String[] signalLabels = reader.readSignalLabels();

        if (signalLabels.length == 0) {
            IJ.error(PLUGIN_NAME, "Could not read signal names from:\n" + lastNanoPath);
            return;
        }

        int defaultIdx = (lastSignalIndex >= 0 && lastSignalIndex < signalLabels.length)
                ? lastSignalIndex
                : reader.defaultSignalIndex();

        if (!showSignalDialog(signalLabels, defaultIdx)) return;

        // ── Main pipeline ────────────────────────────────────────────────────
        try {

            // Step 0: Load EM
            ImagePlus emImage = IJ.openImage(lastEMPath);
            if (emImage == null) {
                IJ.error(PLUGIN_NAME, "Failed to load EM image:\n" + lastEMPath);
                return;
            }

            // Step 0b: Load chosen NanoSIMS channel for registration — first plane only.
            // If the file has multiple planes (z-stack / time-series), only plane 0 is used,
            // matching Python: nano_img = nano_img[:, :, 0]  (3-D) or
            //                  nano_img = nano_img[:, :, 0, signal_idx]  (4-D).
            // The full multi-plane data is applied in Step 3 (AllSignalsWarper).
            ImagePlus nanoImage = reader.readFirstPlaneByIndex(lastSignalIndex);
            if (nanoImage == null) {
                IJ.error(PLUGIN_NAME, "Failed to load NanoSIMS channel: "
                        + signalLabels[lastSignalIndex]);
                return;
            }

            // ── Step 1: Coarse alignment ──────────────────────────────────────
            IJ.showStatus("Step 1: Coarse alignment…");
            IJ.log("EM: " + emResNm + " nm/px   NanoSIMS: " + nanoResNm + " nm/px");

            CoarseAligner coarseAligner = new CoarseAligner(emResNm, nanoResNm);
            CoarseAlignmentResult coarseResult = coarseAligner.align(emImage, nanoImage);

            if (coarseResult == null) {
                IJ.error(PLUGIN_NAME, "Coarse alignment failed.");
                return;
            }
            IJ.log("Coarse done — orientation: " + coarseResult.getOrientationLabel());

            // ── Step 2: Fine alignment (RAFT) ─────────────────────────────────
            IJ.showStatus("Step 2: Fine alignment (RAFT optical flow)…");

            FineAligner fineAligner = new FineAligner();
            FineAlignmentResult fineResult = fineAligner.align(coarseResult);

            if (fineResult == null) {
                IJ.error(PLUGIN_NAME, "Fine alignment failed.");
                return;
            }
            IJ.log("Fine done — method: " + fineResult.getMethod());

            // ── Step 3: Apply warp to all signals and save aligned .nrrd ─────
            IJ.showStatus("Step 3: Warping all signals and saving…");

            String stem      = new File(lastNanoPath).getName().replaceAll("\\.[^.]+$", "");
            String outputDir = new File(lastNanoPath).getParent() + File.separator;
            String outputNrrd = outputDir + stem + "_aligned.nrrd";

            AllSignalsWarper warper = new AllSignalsWarper(
                    lastNanoPath,
                    fineResult.getWarpMatrix(),
                    coarseResult.getOrientationLabel()
            );
            File savedFile = warper.apply(outputNrrd);

            if (savedFile != null)
                IJ.log("Aligned NanoSIMS (nano res) saved to: " + savedFile.getAbsolutePath());
            else
                IJ.log("Warning: could not save aligned NanoSIMS file.");

            // ── Step 4: Extra outputs ─────────────────────────────────────────

            // 4a. Save cropped EM patch at its original (full) EM resolution.
            //     coarseResult.getEMRefinedPatch() is already at original EM px size.
            String emCropPath = outputDir + stem + "_em_crop.tif";
            ImagePlus emCropImp = coarseResult.getEMCrop();
            new FileSaver(emCropImp).saveAsTiff(emCropPath);
            IJ.log("EM crop (original res) saved to: " + emCropPath);

            // 4b. Upsample the warped NanoSIMS from nano resolution to EM resolution.
            //     The EM crop patch has dimensions (coarseResult.getW() × coarseResult.getH())
            //     in EM pixels, which corresponds to the same physical area as the
            //     256×256 nano patch.  Scale factor = nano_res / em_res.
            int emPatchW = coarseResult.getW();
            int emPatchH = coarseResult.getH();

            ImageProcessor warpedNanoProc = fineResult.getAlignedNano().getProcessor();
            ImageProcessor nanoAtEmRes    = warpedNanoProc.resize(emPatchW, emPatchH, true);
            ImagePlus      nanoAtEmResImp = new ImagePlus(
                    stem + "_nano_em_res", nanoAtEmRes);

            // Copy EM calibration (nm/px) so the file carries correct metadata
            ij.measure.Calibration emCal = emCropImp.getCalibration().copy();
            nanoAtEmResImp.setCalibration(emCal);

            String nanoEmResPath = outputDir + stem + "_nano_em_res.tif";
            new FileSaver(nanoAtEmResImp).saveAsTiff(nanoEmResPath);
            IJ.log("Warped NanoSIMS (EM res) saved to: " + nanoEmResPath);

            // ── Display results ───────────────────────────────────────────────
            displayResults(coarseResult, fineResult, nanoAtEmResImp);
            IJ.showStatus("NanoSIMS-EM correlation complete.");

        } catch (Exception e) {
            IJ.error(PLUGIN_NAME, "Error during correlation:\n" + e.getMessage());
            e.printStackTrace();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Dialog 1 — file paths + resolution
    // ─────────────────────────────────────────────────────────────────────────

    private boolean showPathDialog() {
        GenericDialog gd = new GenericDialog(PLUGIN_NAME + " (1/2 — Files)");

        gd.addMessage("=== Input Files ===");
        gd.addStringField("EM Image Path:", lastEMPath, 50);
        gd.addButton("Browse EM…", e -> {
            OpenDialog od = new OpenDialog("Select EM Image", "");
            if (od.getFileName() != null) {
                lastEMPath = od.getDirectory() + od.getFileName();
                ((TextField) gd.getStringFields().get(0)).setText(lastEMPath);
            }
        });

        gd.addStringField("NanoSIMS Path (.im / .nrrd):", lastNanoPath, 50);
        gd.addButton("Browse NanoSIMS…", e -> {
            OpenDialog od = new OpenDialog("Select NanoSIMS File", "");
            if (od.getFileName() != null) {
                lastNanoPath = od.getDirectory() + od.getFileName();
                ((TextField) gd.getStringFields().get(1)).setText(lastNanoPath);
            }
        });

        gd.addMessage("\n=== Resolution Parameters ===");
        gd.addNumericField("EM Resolution (nm/pixel):", emResNm, 2);
        gd.addNumericField("NanoSIMS Resolution (nm/pixel):", nanoResNm, 5);

        gd.showDialog();
        if (gd.wasCanceled()) return false;

        lastEMPath   = gd.getNextString().trim();
        lastNanoPath = gd.getNextString().trim();
        emResNm      = gd.getNextNumber();
        nanoResNm    = gd.getNextNumber();

        if (lastEMPath.isEmpty() || lastNanoPath.isEmpty()) {
            IJ.error(PLUGIN_NAME, "Both file paths must be specified."); return false;
        }
        if (!new File(lastEMPath).exists()) {
            IJ.error(PLUGIN_NAME, "EM image not found:\n" + lastEMPath); return false;
        }
        if (!new File(lastNanoPath).exists()) {
            IJ.error(PLUGIN_NAME, "NanoSIMS file not found:\n" + lastNanoPath); return false;
        }
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Dialog 2 — signal selection
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Shows a dropdown of every mass channel found in the NanoSIMS file and
     * lets the user choose which signal to use for registration.
     * Pre-selects the S32 channel (or the last choice if the file hasn't changed).
     */
    private boolean showSignalDialog(String[] signalLabels, int defaultIdx) {
        GenericDialog gd = new GenericDialog(PLUGIN_NAME + " (2/2 — Registration Signal)");

        gd.addMessage("Choose the NanoSIMS signal to use for EM registration.\n"
                + "Strong, structured signals (e.g. 32S, 12C14N) work best.");

        gd.addChoice("Registration signal:", signalLabels, signalLabels[defaultIdx]);

        gd.showDialog();
        if (gd.wasCanceled()) return false;

        String chosen = gd.getNextChoice();
        // Find its index back from the label list
        lastSignalIndex = defaultIdx; // fallback
        for (int i = 0; i < signalLabels.length; i++) {
            if (signalLabels[i].equals(chosen)) { lastSignalIndex = i; break; }
        }
        lastSignalLabel = chosen;

        IJ.log("Registration signal: [" + lastSignalIndex + "] " + lastSignalLabel);
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Results display
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @param nanoAtEmRes  warped NanoSIMS already upsampled to EM patch resolution
     */
    private void displayResults(CoarseAlignmentResult coarse, FineAlignmentResult fine,
                                ImagePlus nanoAtEmRes) {
        int w = fine.getEMBoundaries().getWidth();
        int h = fine.getEMBoundaries().getHeight();

        // ── Panel at nano resolution (unchanged) ──────────────────────────────
        ImageStack stack = new ImageStack(w, h);
        stack.addSlice("1_EM_Refined_Patch",  safeResize(coarse.getEMCrop(),    w, h));
        stack.addSlice("2_Nano_Preprocessed", safeResize(coarse.getNanoImage(), w, h));
        stack.addSlice("3_EM_Contrast",       fine.getEMBoundaries().getProcessor());
        stack.addSlice("4_Nano_Contrast",     fine.getNanoBoundaries().getProcessor());
        stack.addSlice("5_Nano_Warped",       fine.getAlignedNano().getProcessor());
        stack.addSlice("6_Overlay",           fine.getOverlay().getProcessor());

        new ImagePlus("Correlation Results [" + lastSignalLabel + "]", stack).show();

        // ── High-res colour overlay at EM resolution (displayed, not saved) ──
        //    EM base = original EM crop patch (full EM px resolution)
        //    Nano layer = warped nano upsampled to same EM patch size
        ImagePlus emHighResImp = coarse.getEMCrop();
        ImagePlus overlayHiRes = createColorOverlay(emHighResImp, nanoAtEmRes);
        overlayHiRes.setTitle("Final EM-NanoSIMS Overlay (EM res) [" + lastSignalLabel + "]");
        overlayHiRes.show();

        double[][] M = fine.getWarpMatrix();
        IJ.log(String.format("Warp matrix:\n  [%.5f  %.5f  %.3f]\n  [%.5f  %.5f  %.3f]",
                M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2]));
    }

    private ImageProcessor safeResize(ImagePlus imp, int w, int h) {
        if (imp.getWidth() == w && imp.getHeight() == h) return imp.getProcessor();
        return imp.getProcessor().resize(w, h);
    }

    /**
     * Builds a colour overlay matching Python's final_overlay_hires.png:
     *   - EM displayed as gray base
     *   - NanoSIMS mapped through the 'hot' colormap as a semi-transparent RGBA layer
     *   - alpha = clip(nano / 0.04, 0, 1) * 0.55  (intensity-driven, no feather mask)
     */
    private ImagePlus createColorOverlay(ImagePlus emImp, ImagePlus nanoImp) {
        int w = emImp.getWidth();
        int h = emImp.getHeight();

        ImageProcessor emProc   = emImp.getProcessor().convertToByteProcessor();
        ImageProcessor nanoProc = nanoImp.getProcessor().resize(w, h).convertToByteProcessor();

        ColorProcessor overlay = new ColorProcessor(w, h);

        final float MAX_ALPHA        = 0.55f;
        final float INTENSITY_THRESH = 0.04f;  // intensity_thresh * 4  (Python)

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float em   = emProc.get(x, y)  / 255f;
                float nano = nanoProc.get(x, y) / 255f;

                // Gray EM base
                float base = em;

                // Hot colormap: 0→black  0.33→red  0.67→yellow  1→white
                float hr, hg, hb;
                if (nano < 1f/3f) {
                    float t = nano / (1f/3f);
                    hr = t;  hg = 0f; hb = 0f;
                } else if (nano < 2f/3f) {
                    float t = (nano - 1f/3f) / (1f/3f);
                    hr = 1f; hg = t;  hb = 0f;
                } else {
                    float t = (nano - 2f/3f) / (1f/3f);
                    hr = 1f; hg = 1f; hb = t;
                }

                float alpha = Math.min(1f, nano / INTENSITY_THRESH) * MAX_ALPHA;

                int r = Math.min(255, (int)((1f - alpha) * base * 255 + alpha * hr * 255));
                int g = Math.min(255, (int)((1f - alpha) * base * 255 + alpha * hg * 255));
                int b = Math.min(255, (int)((1f - alpha) * base * 255 + alpha * hb * 255));

                overlay.putPixel(x, y, new int[]{r, g, b});
            }
        }
        return new ImagePlus("Final Overlay (hot)", overlay);
    }
}
