package com.haibolab.nansimsemcorrelation;

import ij.ImagePlus;
import ij.process.ImageProcessor;

/**
 * Holds the result of the RAFT-based fine alignment step.
 *
 * Slot mapping (aligned with Python demo() output):
 *   alignedNano   — NanoSIMS patch after warpAffine
 *   emPreproc     — contrast-adjusted EM patch (used as "emBoundaries" display slot)
 *   nanoPreproc   — contrast-adjusted NanoSIMS patch (used as "nanoBoundaries" display slot)
 *   overlay       — simple EM+Nano composite for quick review
 *   warpMatrix    — 2×3 affine matrix estimated by RANSAC
 *   method        — "RAFT"
 */
public class FineAlignmentResult {
    private final ImageProcessor alignedNano;
    private final ImageProcessor emPreproc;
    private final ImageProcessor nanoPreproc;
    private final ImageProcessor overlay;
    private final double[][] warpMatrix;
    private final String method;

    public FineAlignmentResult(ImageProcessor aligned, ImageProcessor emPreproc,
                               ImageProcessor nanoPreproc, ImageProcessor overlay,
                               double[][] matrix, String method) {
        this.alignedNano  = aligned;
        this.emPreproc    = emPreproc;
        this.nanoPreproc  = nanoPreproc;
        this.overlay      = overlay;
        this.warpMatrix   = matrix;
        this.method       = method;
    }

    public ImagePlus getAlignedNano()   { return new ImagePlus("Aligned Nano",  alignedNano);  }
    /** Preprocessed (contrast-adjusted) EM patch – used as EM reference for overlay. */
    public ImagePlus getEMBoundaries()  { return new ImagePlus("EM Preprocessed", emPreproc); }
    /** Preprocessed (contrast-adjusted) NanoSIMS patch – used as before-warp reference. */
    public ImagePlus getNanoBoundaries(){ return new ImagePlus("Nano Preprocessed", nanoPreproc); }
    public ImagePlus getOverlay()       { return new ImagePlus("Overlay", overlay);            }
    public double[][] getWarpMatrix()   { return warpMatrix;  }
    public String getMethod()           { return method;       }
}
