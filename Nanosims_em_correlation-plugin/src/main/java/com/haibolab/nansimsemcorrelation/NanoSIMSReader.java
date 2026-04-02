package com.haibolab.nansimsemcorrelation;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import java.io.*;
import com.nrims.data.Mims_Reader;
import com.nrims.data.Nrrd_Reader;
import com.nrims.data.Opener;

public class NanoSIMSReader {
    private final String filePath;
    private Opener reader;
    private boolean isNrrdFile;

    public NanoSIMSReader(String path) {
        this.filePath = path;
        this.isNrrdFile = path.toLowerCase().endsWith(".nrrd");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API: signal discovery
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Returns display labels for every mass channel in the file,
     * e.g. ["32S (32.00)", "31P (30.97)", "12C (12.00)"].
     * Safe to call without keeping the reader open — opens and closes internally.
     */
    public String[] readSignalLabels() {
        try {
            openReader();
            String[] names   = reader.getMassNames();
            String[] symbols = reader.getMassSymbols();
            int n = (names != null) ? names.length : 0;
            String[] labels = new String[n];
            for (int i = 0; i < n; i++) {
                String sym  = (symbols != null && i < symbols.length) ? symbols[i].trim() : "";
                String name = (names[i] != null) ? names[i].trim() : ("ch" + i);
                labels[i] = sym.isEmpty() || sym.equals("-")
                        ? name
                        : sym + " (" + name + ")";
            }
            return labels;
        } catch (Exception e) {
            IJ.log("Warning: could not read signal names from " + filePath + ": " + e.getMessage());
            return new String[0];
        } finally {
            closeReader();
        }
    }

    /**
     * Returns the index of the first channel whose symbol matches one of the
     * common S32 identifiers, or 0 if none found (safe default).
     */
    public int defaultSignalIndex() {
        try {
            openReader();
            int idx = findS32ChannelIndex();
            return Math.max(0, idx);
        } catch (Exception e) {
            return 0;
        } finally {
            closeReader();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API: reading channels
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Reads the S32 channel (auto-detected) and returns it as an ImagePlus.
     * Kept for backward compatibility.
     */
    public ImagePlus readS32Channel() {
        try {
            openReader();
            int s32Index = findS32ChannelIndex();
            if (s32Index < 0) {
                IJ.error("S32 channel not found in file: " + filePath);
                return null;
            }
            ImagePlus imp = readChannel(s32Index);
            if (imp != null) imp.setTitle("S32_" + new File(filePath).getName());
            return imp;
        } catch (Exception e) {
            IJ.error("Error reading NanoSIMS file", e.getMessage());
            return null;
        } finally {
            closeReader();
        }
    }

    /**
     * Reads any channel by index and returns it as an ImagePlus (all planes).
     * The title is set to the channel's mass symbol.
     *
     * @param channelIndex 0-based index into the mass channels
     */
    public ImagePlus readChannelByIndex(int channelIndex) {
        try {
            openReader();
            int nMasses = reader.getNMasses();
            if (channelIndex < 0 || channelIndex >= nMasses) {
                IJ.error("Channel index " + channelIndex + " out of range (0–" + (nMasses-1) + ")");
                return null;
            }
            String[] symbols = reader.getMassSymbols();
            String title = (symbols != null && channelIndex < symbols.length)
                    ? symbols[channelIndex].trim()
                    : ("ch" + channelIndex);
            ImagePlus imp = readChannel(channelIndex);
            if (imp != null) imp.setTitle(title + "_" + new File(filePath).getName());
            return imp;
        } catch (Exception e) {
            IJ.error("Error reading NanoSIMS channel " + channelIndex, e.getMessage());
            return null;
        } finally {
            closeReader();
        }
    }

    /**
     * Reads only the <b>first plane</b> of a mass channel for use in
     * registration. If the file has multiple planes (z-stack / time-series),
     * only plane 0 is loaded — matching Python:
     * {@code nano_img = nano_img[:, :, 0]}  (3-D) or
     * {@code nano_img = nano_img[:, :, 0, signal_idx]}  (4-D).
     *
     * @param channelIndex 0-based mass channel index
     * @return single-frame ImagePlus, or null on error
     */
    public ImagePlus readFirstPlaneByIndex(int channelIndex) {
        try {
            openReader();
            int nMasses = reader.getNMasses();
            if (channelIndex < 0 || channelIndex >= nMasses) {
                IJ.error("Channel index " + channelIndex + " out of range (0–" + (nMasses-1) + ")");
                return null;
            }

            int nPlanes = reader.getNImages();
            if (nPlanes > 1) {
                IJ.log("NanoSIMS channel has " + nPlanes + " planes — "
                        + "using plane 1 for registration (matching Python behaviour).");
            }

            // Read only plane 0
            reader.setStackIndex(0);
            Object pixels = reader.getPixels(channelIndex);
            if (pixels == null) {
                IJ.error("Failed to read plane 1 of channel " + channelIndex);
                return null;
            }

            int width  = reader.getWidth();
            int height = reader.getHeight();
            ImageProcessor ip;
            if (pixels instanceof short[]) {
                ShortProcessor sp = new ShortProcessor(width, height);
                sp.setPixels(pixels);
                ip = sp;
            } else if (pixels instanceof float[]) {
                FloatProcessor fp = new FloatProcessor(width, height);
                fp.setPixels(pixels);
                ip = fp;
            } else if (pixels instanceof int[]) {
                // convert int[] → float
                int[] raw = (int[]) pixels;
                float[] fdata = new float[raw.length];
                for (int i = 0; i < raw.length; i++) fdata[i] = raw[i];
                ip = new FloatProcessor(width, height, fdata, null);
            } else if (pixels instanceof byte[]) {
                ip = new ByteProcessor(width, height, (byte[]) pixels, null);
            } else {
                IJ.error("Unsupported pixel type: " + pixels.getClass().getName());
                return null;
            }

            String[] symbols = reader.getMassSymbols();
            String title = (symbols != null && channelIndex < symbols.length
                    && !symbols[channelIndex].equals("-"))
                    ? symbols[channelIndex].trim()
                    : ("ch" + channelIndex);

            ImagePlus imp = new ImagePlus(title + "_plane1_" + new File(filePath).getName(), ip);

            float pw = reader.getPixelWidth(), ph = reader.getPixelHeight();
            if (pw > 0 && ph > 0) {
                ij.measure.Calibration cal = imp.getCalibration();
                cal.pixelWidth = pw; cal.pixelHeight = ph; cal.setUnit("um");
            }
            return imp;

        } catch (Exception e) {
            IJ.error("Error reading NanoSIMS first plane for channel " + channelIndex,
                    e.getMessage());
            return null;
        } finally {
            closeReader();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void openReader() throws IOException {
        File file = new File(filePath);
        if (isNrrdFile) {
            reader = new Nrrd_Reader(file);
        } else {
            reader = new Mims_Reader(file);
        }
    }

    private void closeReader() {
        if (reader != null) { reader.close(); reader = null; }
    }

    private int findS32ChannelIndex() {
        String[] massNames   = reader.getMassNames();
        String[] massSymbols = reader.getMassSymbols();
        if (massNames == null) return -1;
        for (int i = 0; i < massNames.length; i++) {
            String name   = massNames[i].trim();
            String symbol = (massSymbols != null && i < massSymbols.length)
                    ? massSymbols[i].trim() : "";
            if (isS32Mass(name, symbol)) return i;
        }
        return -1;
    }

    private boolean isS32Mass(String massName, String massSymbol) {
        try {
            double mass = Double.parseDouble(massName);
            if (Math.abs(mass - 32.0) < 0.1) {
                if (massSymbol.isEmpty() || massSymbol.equals("-")) return true;
                if (massSymbol.toUpperCase().contains("S")) return true;
            }
        } catch (NumberFormatException ignored) {}
        String su = massSymbol.toUpperCase();
        return su.equals("32S") || su.equals("S32")
                || (su.equals("S") && massName.startsWith("32"));
    }

    private ImagePlus readChannel(int channelIndex) throws IOException {
        int width   = reader.getWidth();
        int height  = reader.getHeight();
        int nImages = reader.getNImages();
        ImageStack stack = new ImageStack(width, height);
        for (int plane = 0; plane < nImages; plane++) {
            reader.setStackIndex(plane);
            Object pixels = reader.getPixels(channelIndex);
            if (pixels == null) {
                IJ.error("Failed to read plane " + (plane+1) + " of channel " + channelIndex);
                return null;
            }
            if (pixels instanceof short[]) {
                ShortProcessor sp = new ShortProcessor(width, height);
                sp.setPixels(pixels);
                stack.addSlice("Plane_" + (plane+1), sp);
            } else if (pixels instanceof float[]) {
                FloatProcessor fp = new FloatProcessor(width, height);
                fp.setPixels(pixels);
                stack.addSlice("Plane_" + (plane+1), fp);
            } else {
                IJ.error("Unsupported pixel type");
                return null;
            }
        }
        ImagePlus imp = new ImagePlus("ch" + channelIndex, stack);
        float pw = reader.getPixelWidth(), ph = reader.getPixelHeight();
        if (pw > 0 && ph > 0) {
            ij.measure.Calibration cal = imp.getCalibration();
            cal.pixelWidth = pw; cal.pixelHeight = ph; cal.setUnit("um");
        }
        return imp;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Metadata helpers (kept for legacy getFileInfo() callers)
    // ─────────────────────────────────────────────────────────────────────────

    public String getFileInfo() {
        try {
            openReader();
            StringBuilder info = new StringBuilder();
            info.append("File: ").append(filePath).append("\n");
            info.append("Dimensions: ").append(reader.getWidth()).append(" x ")
                    .append(reader.getHeight()).append("\n");
            info.append("Planes: ").append(reader.getNImages()).append("\n");
            info.append("Masses: ").append(reader.getNMasses()).append("\n");
            String[] names   = reader.getMassNames();
            String[] symbols = reader.getMassSymbols();
            for (int i = 0; i < names.length; i++) {
                info.append("  [").append(i).append("] ").append(names[i]);
                if (symbols != null && i < symbols.length && !symbols[i].equals("-"))
                    info.append(" (").append(symbols[i]).append(")");
                info.append("\n");
            }
            return info.toString();
        } catch (Exception e) {
            return "Error reading file: " + e.getMessage();
        } finally {
            closeReader();
        }
    }
}
