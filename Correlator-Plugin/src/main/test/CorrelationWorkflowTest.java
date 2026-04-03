import com.haibolab.nansimsemcorrelation.*;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import org.junit.Before;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.*;

public class CorrelationWorkflowTest {

    // Test data paths - UPDATE THESE TO MATCH YOUR DATA
    private static final String EM_PATH = "/Users/luchixiang/Downloads/NanoSIMS EM correlation/Tissue （map）/Xu liver/Region 1.tif";
    private static final String NANO_PATH = "/Users/luchixiang/Downloads/NanoSIMS EM correlation/Tissue （map）/Xu liver/mosaic.nrrd";
    // NOTE: SAM checkpoint no longer needed — fine alignment uses the bundled RAFT model.

    // Resolution parameters
    private static final double EM_RES_NM = 6.0;
    private static final double NANO_RES_NM = 97.65625;

    // Output directory for debugging
    private static final String OUTPUT_DIR = System.getProperty("user.home") + "/correlation_debug/";

    private ImageJ ij;

    @Before
    public void setUp() {
        // Initialize ImageJ
        if (ij == null) {
            ij = new ImageJ();
        }

        // Create output directory
        new File(OUTPUT_DIR).mkdirs();

        System.out.println("╔════════════════════════════════════════════════════════╗");
        System.out.println("║  NanoSIMS-EM Correlation Workflow Test               ║");
        System.out.println("╚════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    @Test
    public void testFullCorrelationWorkflow() {
        System.out.println("=== Testing Full Correlation Workflow ===\n");

        // Validate input files
        assertTrue("EM image file must exist: " + EM_PATH,
                new File(EM_PATH).exists());
        assertTrue("NanoSIMS file must exist: " + NANO_PATH,
                new File(NANO_PATH).exists());

        try {
            // Step 1: Load and validate images
            System.out.println("--- Step 0: Loading Images ---");
            ImagePlus emImage = loadAndValidateEM(EM_PATH);
            ImagePlus nanoImage = loadAndValidateNanoSIMS(NANO_PATH);

            // Step 2: Coarse alignment
            System.out.println("\n--- Step 1: Coarse Alignment ---");
            CoarseAlignmentResult step1Result = performCoarseAlignment(emImage, nanoImage);

            // Step 3: Fine alignment with SAM
            System.out.println("\n--- Step 2: Fine Alignment with SAM ---");
            FineAlignmentResult step2Result = performFineAlignment(step1Result);

            // Step 4: Validate and save results
            System.out.println("\n--- Step 3: Validating Results ---");
            validateAndSaveResults(step1Result, step2Result);

            System.out.println("\n✅ Correlation workflow completed successfully!");
            System.out.println("📁 Results saved to: " + OUTPUT_DIR);

        } catch (Exception e) {
            fail("Correlation workflow failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Test
    public void testStep1Only() {
        System.out.println("=== Testing Step 1 (Coarse Alignment) Only ===\n");

        if (!new File(EM_PATH).exists() || !new File(NANO_PATH).exists()) {
            System.out.println("⚠ Skipping test - input files not found");
            return;
        }

        try {
            ImagePlus emImage = IJ.openImage(EM_PATH);
            NanoSIMSReader reader = new NanoSIMSReader(NANO_PATH);
            ImagePlus nanoImage = reader.readS32Channel();

            assertNotNull("EM image should load", emImage);
            assertNotNull("NanoSIMS image should load", nanoImage);

            CoarseAligner aligner = new CoarseAligner(EM_RES_NM, NANO_RES_NM);
            CoarseAlignmentResult result = aligner.align(emImage, nanoImage);

            assertNotNull("Step 1 result should not be null", result);
            assertNotNull("EM crop should not be null", result.getEMCrop());
            assertNotNull("Aligned nano should not be null", result.getNanoImage());

            // Save intermediate results
            IJ.save(result.getEMCrop(), OUTPUT_DIR + "step1_em_crop.tif");
            IJ.save(result.getNanoImage(), OUTPUT_DIR + "step1_nano_aligned.tif");

            System.out.println("✅ Step 1 completed");
            System.out.println("  Orientation: " + result.getOrientationLabel());
            System.out.println("  Crop coords: (" + result.getX() + "," + result.getY() +
                    "," + result.getW() + "," + result.getH() + ")");

        } catch (Exception e) {
            fail("Step 1 failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Test
    public void testImageQuality() {
        System.out.println("=== Testing Image Quality Metrics ===\n");

        if (!new File(EM_PATH).exists() || !new File(NANO_PATH).exists()) {
            System.out.println("⚠ Skipping test - input files not found");
            return;
        }

        try {
            ImagePlus emImage = IJ.openImage(EM_PATH);
            NanoSIMSReader reader = new NanoSIMSReader(NANO_PATH);
            ImagePlus nanoImage = reader.readS32Channel();

            // Check EM quality
            ImageProcessor emProc = emImage.getProcessor();
            double emMean = emProc.getStatistics().mean;
            double emStdDev = emProc.getStatistics().stdDev;
            double emContrast = emStdDev / emMean;

            System.out.println("EM Image Quality:");
            System.out.println("  Size: " + emImage.getWidth() + "x" + emImage.getHeight());
            System.out.println("  Mean: " + String.format("%.2f", emMean));
            System.out.println("  StdDev: " + String.format("%.2f", emStdDev));
            System.out.println("  Contrast: " + String.format("%.2f", emContrast));

            assertTrue("EM contrast should be reasonable", emContrast > 0.1);

            // Check NanoSIMS quality
            ImageProcessor nanoProc = nanoImage.getProcessor();
            double nanoMean = nanoProc.getStatistics().mean;
            double nanoStdDev = nanoProc.getStatistics().stdDev;
            double nanoContrast = nanoStdDev / nanoMean;

            System.out.println("\nNanoSIMS Image Quality:");
            System.out.println("  Size: " + nanoImage.getWidth() + "x" + nanoImage.getHeight());
            System.out.println("  Mean: " + String.format("%.2f", nanoMean));
            System.out.println("  StdDev: " + String.format("%.2f", nanoStdDev));
            System.out.println("  Contrast: " + String.format("%.2f", nanoContrast));

            assertTrue("NanoSIMS contrast should be reasonable", nanoContrast > 0.05);

        } catch (Exception e) {
            fail("Image quality test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Test
    public void testAlignmentMetrics() {
        System.out.println("=== Testing Alignment Quality Metrics ===\n");

        if (!new File(EM_PATH).exists() || !new File(NANO_PATH).exists()) {
            System.out.println("⚠ Skipping test - input files not found");
            return;
        }

        try {
            ImagePlus emImage = IJ.openImage(EM_PATH);
            NanoSIMSReader reader = new NanoSIMSReader(NANO_PATH);
            ImagePlus nanoImage = reader.readS32Channel();

            CoarseAligner aligner = new CoarseAligner(EM_RES_NM, NANO_RES_NM);
            CoarseAlignmentResult result = aligner.align(emImage, nanoImage);

            // Calculate alignment quality
            ImageProcessor emCrop = result.getEMRefinedPatch();
            ImageProcessor nanoAligned = result.getNanoAligned();

            // Resize to same size for comparison
            int targetW = Math.min(emCrop.getWidth(), nanoAligned.getWidth());
            int targetH = Math.min(emCrop.getHeight(), nanoAligned.getHeight());

            ImageProcessor emResized = emCrop.resize(targetW, targetH);
            ImageProcessor nanoResized = nanoAligned.resize(targetW, targetH);

            // Calculate correlation
            double correlation = calculateNCC(emResized, nanoResized);

            System.out.println("Alignment Quality Metrics:");
            System.out.println("  Normalized Cross-Correlation: " + String.format("%.4f", correlation));
            System.out.println("  Orientation: " + result.getOrientationLabel());

            assertTrue("Correlation should be positive", correlation > 0);

            // Calculate overlap metrics
            int overlapPixels = calculateOverlapPixels(emResized, nanoResized);
            double overlapRatio = (double)overlapPixels / (targetW * targetH);

            System.out.println("  Overlap ratio: " + String.format("%.2f%%", overlapRatio * 100));

        } catch (Exception e) {
            fail("Alignment metrics test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // ========== Helper Methods ==========

    private ImagePlus loadAndValidateEM(String path) {
        System.out.println("Loading EM image: " + path);
        ImagePlus imp = IJ.openImage(path);

        assertNotNull("EM image should load successfully", imp);
        assertTrue("EM width should be positive", imp.getWidth() > 0);
        assertTrue("EM height should be positive", imp.getHeight() > 0);

        System.out.println("  ✓ EM loaded: " + imp.getWidth() + "x" + imp.getHeight());
        return imp;
    }

    private ImagePlus loadAndValidateNanoSIMS(String path) {
        System.out.println("Loading NanoSIMS: " + path);

        NanoSIMSReader reader = new NanoSIMSReader(path);
        ImagePlus imp = reader.readS32Channel();

        assertNotNull("NanoSIMS image should load successfully", imp);
        assertTrue("NanoSIMS width should be positive", imp.getWidth() > 0);
        assertTrue("NanoSIMS height should be positive", imp.getHeight() > 0);

        System.out.println("  ✓ NanoSIMS loaded: " + imp.getWidth() + "x" + imp.getHeight());
        return imp;
    }

    private CoarseAlignmentResult performCoarseAlignment(ImagePlus emImage, ImagePlus nanoImage) {
        CoarseAligner aligner = new CoarseAligner(EM_RES_NM, NANO_RES_NM);
        CoarseAlignmentResult result = aligner.align(emImage, nanoImage);

        assertNotNull("Coarse alignment result should not be null", result);
        assertNotNull("EM refined patch should not be null", result.getEMRefinedPatch());
        assertNotNull("Nano aligned should not be null", result.getNanoAligned());
        ImagePlus em_image = result.getEMCrop();
//        IJ.run(em_image, "Apply LUT", "");  // <-- Add this line
        // Save intermediate results
        IJ.save(em_image, OUTPUT_DIR + "1_em_crop.tif");
        ImagePlus nano_image = result.getNanoImage();
//        IJ.run(nano_image, "Apply LUT", "");

        IJ.save(nano_image, OUTPUT_DIR + "2_nano_aligned.tif");

        System.out.println("  ✓ Coarse alignment completed");
        System.out.println("    Orientation: " + result.getOrientationLabel());
        System.out.println("    Crop region: x=" + result.getX() + ", y=" + result.getY() +
                ", w=" + result.getW() + ", h=" + result.getH());

        return result;
    }

    private FineAlignmentResult performFineAlignment(CoarseAlignmentResult coarseResult) {
        // RAFT model is loaded from the bundled resource — no path argument needed
        FineAligner aligner = new FineAligner();
        FineAlignmentResult result = aligner.align(coarseResult);

        assertNotNull("Fine alignment result should not be null", result);
        assertNotNull("Aligned nano should not be null", result.getAlignedNano());
        assertNotNull("EM boundaries should not be null", result.getEMBoundaries());
        assertNotNull("Nano boundaries should not be null", result.getNanoBoundaries());

        // Save intermediate results
        IJ.save(result.getEMBoundaries(), OUTPUT_DIR + "3_em_boundaries.tif");
        IJ.save(result.getNanoBoundaries(), OUTPUT_DIR + "4_nano_boundaries.tif");
        IJ.save(result.getAlignedNano(), OUTPUT_DIR + "5_aligned_nano.tif");
        IJ.save(result.getOverlay(), OUTPUT_DIR + "6_overlay.tif");

        System.out.println("  ✓ Fine alignment completed");
        System.out.println("    Method: " + result.getMethod());

        return result;
    }

    private void validateAndSaveResults(CoarseAlignmentResult step1, FineAlignmentResult step2) {
        // Create final overlay
        ImageProcessor emCrop = step1.getEMRefinedPatch();
        ImageProcessor nanoFinal = step2.getAlignedNano().getProcessor();

        // Resize to match
        int w = emCrop.getWidth();
        int h = emCrop.getHeight();
        ImageProcessor nanoResized = nanoFinal.resize(w, h);

        // Create color overlay
        ImagePlus overlay = createColorOverlay(
                new ImagePlus("EM", emCrop),
                new ImagePlus("Nano", nanoResized)
        );

        IJ.save(overlay, OUTPUT_DIR + "7_final_overlay.tif");

        // Calculate final quality metrics
        double finalNCC = calculateNCC(emCrop.resize(256, 256), nanoResized.resize(256, 256));

        System.out.println("  Final NCC: " + String.format("%.4f", finalNCC));

        assertTrue("Final alignment quality should be reasonable", finalNCC > -0.5);

        // Save warp matrix
        double[][] warpMatrix = step2.getWarpMatrix();
        System.out.println("  Warp matrix:");
        for (int i = 0; i < warpMatrix.length; i++) {
            System.out.print("    [");
            for (int j = 0; j < warpMatrix[i].length; j++) {
                System.out.print(String.format("%.4f", warpMatrix[i][j]));
                if (j < warpMatrix[i].length - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }

    private ImagePlus createColorOverlay(ImagePlus em, ImagePlus nano) {
        int w = em.getWidth();
        int h = em.getHeight();

        ImageProcessor emProc = em.getProcessor().convertToByteProcessor();
        ImageProcessor nanoProc = nano.getProcessor().resize(w, h).convertToByteProcessor();

        ij.process.ColorProcessor overlay = new ij.process.ColorProcessor(w, h);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int emVal = emProc.get(x, y);
                int nanoVal = nanoProc.get(x, y);

                // EM in grayscale, NanoSIMS in red-yellow
                int r = Math.min(255, emVal + nanoVal);
                int g = Math.min(255, emVal + nanoVal / 2);
                int b = emVal;

                overlay.putPixel(x, y, new int[]{r, g, b});
            }
        }

        return new ImagePlus("Overlay", overlay);
    }

    private double calculateNCC(ImageProcessor img1, ImageProcessor img2) {
        int w = Math.min(img1.getWidth(), img2.getWidth());
        int h = Math.min(img1.getHeight(), img2.getHeight());

        double sum1 = 0, sum2 = 0;
        double sum1Sq = 0, sum2Sq = 0;
        double sumProduct = 0;
        int count = w * h;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double v1 = img1.getPixel(x, y);
                double v2 = img2.getPixel(x, y);

                sum1 += v1;
                sum2 += v2;
                sum1Sq += v1 * v1;
                sum2Sq += v2 * v2;
                sumProduct += v1 * v2;
            }
        }

        double mean1 = sum1 / count;
        double mean2 = sum2 / count;

        double numerator = sumProduct - count * mean1 * mean2;
        double denom1 = Math.sqrt(sum1Sq - count * mean1 * mean1);
        double denom2 = Math.sqrt(sum2Sq - count * mean2 * mean2);

        if (denom1 == 0 || denom2 == 0) return 0;

        return numerator / (denom1 * denom2);
    }

    private int calculateOverlapPixels(ImageProcessor img1, ImageProcessor img2) {
        int count = 0;
        int w = Math.min(img1.getWidth(), img2.getWidth());
        int h = Math.min(img1.getHeight(), img2.getHeight());

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (img1.getPixel(x, y) > 0 && img2.getPixel(x, y) > 0) {
                    count++;
                }
            }
        }

        return count;
    }
}
