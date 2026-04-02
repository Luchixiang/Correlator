package com.haibolab.nansimsemcorrelation;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Wraps the RAFT DJL model for use in the EM-NanoSIMS correlator.
 *
 * The model is loaded from the bundled resource "raft_modelv3.2.zip"
 * (same approach as Util.getResourcePath in NanoSIMS_Stabilizer_Plugin).
 *
 * Input:  two NDArrays of shape [H, W] (grayscale float)
 * Output: float[] of shape [2 * H * W]  —  [u0..uN, v0..vN]
 *         (RAFT flow, channel-first, i.e. [2, H, W] flattened)
 */
public class RaftFlowEstimator implements AutoCloseable {

    private static final String RESOURCE_NAME = "raft_modelv3.2.zip";

    private final Model model;

    /**
     * Loads the RAFT model from the bundled "raft_modelv3.2.zip" resource.
     * The zip is extracted to a temp file so DJL can load it — identical to
     * how the NanoSIMS Stabilizer plugin loads its model.
     */
    public RaftFlowEstimator() throws IOException, MalformedModelException {
        Path modelPath = extractResourceToTemp(RESOURCE_NAME);
        this.model = Model.newInstance("RAFT");
        this.model.load(modelPath);
    }

    /**
     * Extracts a classpath resource to a temporary file and returns its path.
     * Mirrors Util.getResourcePath() from the Stabilizer plugin.
     */
    private static Path extractResourceToTemp(String resourceName) throws IOException {
        try (InputStream in = RaftFlowEstimator.class
                .getClassLoader().getResourceAsStream(resourceName)) {
            if (in == null) {
                throw new IllegalArgumentException(
                        "Bundled resource not found on classpath: " + resourceName);
            }
            Path tmp = Files.createTempFile("raft_model_", ".zip");
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            return tmp;
        }
    }

    /**
     * Estimate optical flow from image1 → image2.
     *
     * @param image1 reference image (e.g. EM), NDArray [H, W] float
     * @param image2 moving image (e.g. NanoSIMS), NDArray [H, W] float
     * @return flow as float[2*H*W] (channel-first: [u, v] flattened)
     */
    public float[] estimateFlow(NDArray image1, NDArray image2)
            throws ModelException, TranslateException, IOException {

        Translator<Pair<NDArray, NDArray>, float[]> translator =
                new RaftTranslator();

        try (Predictor<Pair<NDArray, NDArray>, float[]> predictor =
                     model.newPredictor(translator, Device.cpu())) {
            return predictor.predict(new Pair<>(image1, image2));
        }
    }

    @Override
    public void close() {
        if (model != null) {
            model.close();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inner translator — mirrors OpticalFlowTranslatorArray from Stabilizer
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Translates a pair of [H,W] grayscale float NDArrays into the RAFT
     * model's input format and unpacks the flow output.
     *
     * Input pre-processing (matching Python load_image2 / OpticalFlowTranslatorArray):
     *   - stack grayscale → 3-channel  (expandDims then repeat along new axis)
     *   - shape: [H, W] → [1, 3, H, W]
     */
    static class RaftTranslator implements Translator<Pair<NDArray, NDArray>, float[]> {

        @Override
        public NDList processInput(TranslatorContext ctx, Pair<NDArray, NDArray> input) {
            // Each image: [H, W] → [H, W, 1] → [H, W, 3] → [3, H, W]
            // (matches OpticalFlowTranslatorArray: expandDims(2).repeat(2,3).transpose(2,0,1))
            NDArray img1 = input.getKey()
                    .expandDims(2)
                    .repeat(2, 3)
                    .transpose(2, 0, 1)
                    .toType(DataType.FLOAT32, false);

            NDArray img2 = input.getValue()
                    .expandDims(2)
                    .repeat(2, 3)
                    .transpose(2, 0, 1)
                    .toType(DataType.FLOAT32, false);

            NDList list = new NDList();
            list.add(0, img1);
            list.add(1, img2);
            return list;
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            // Output: flow tensor [1, 2, H, W] → flatten to float[]
            NDArray flowArray = list.get(0);
            return flowArray.toFloatArray();
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}

