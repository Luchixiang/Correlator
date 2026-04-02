package com.haibolab.nansimsemcorrelation;

import ij.ImagePlus;
import ij.process.ImageProcessor;

public class CoarseAlignmentResult {
    private final ImagePlus emHighRes;
    private final ImageProcessor emRefinedPatch;
    private final ImageProcessor nanoAligned;
    private final int x, y, w, h;
    private final String orientationLabel;

    public CoarseAlignmentResult(ImagePlus emHighRes, ImageProcessor emPatch,
                                 ImageProcessor nanoAligned, int x, int y, int w, int h,
                                 String label) {
        this.emHighRes = emHighRes;
        this.emRefinedPatch = emPatch;
        this.nanoAligned = nanoAligned;
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.orientationLabel = label;
    }

    public ImagePlus getEMHighRes() { return emHighRes; }
    public ImageProcessor getEMRefinedPatch() { return emRefinedPatch; }
    public ImageProcessor getNanoAligned() { return nanoAligned; }
    public int getX() { return x; }
    public int getY() { return y; }
    public int getW() { return w; }
    public int getH() { return h; }
    public String getOrientationLabel() { return orientationLabel; }

    public ImagePlus getEMCrop() {
        return new ImagePlus("EM Crop", emRefinedPatch);
    }

    public ImagePlus getNanoImage() {
        return new ImagePlus("Nano Aligned", nanoAligned);
    }
}
