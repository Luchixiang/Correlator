package com.haibolab.nansimsemcorrelation;

import com.haibolab.nansimsemcorrelation.TemplateMatchResult;
import ij.process.ImageProcessor;

public class TemplateMatching {

    public static TemplateMatchResult match(ImageProcessor image, ImageProcessor template) {
        int imgW = image.getWidth();
        int imgH = image.getHeight();
        int tmpW = template.getWidth();
        int tmpH = template.getHeight();

        if (imgW < tmpW || imgH < tmpH) {
            return new TemplateMatchResult(-1, 0, 0);
        }

        double bestScore = -Double.MAX_VALUE;
        int bestX = 0;
        int bestY = 0;

        // Normalized Cross-Correlation
        for (int y = 0; y <= imgH - tmpH; y++) {
            for (int x = 0; x <= imgW - tmpW; x++) {
                double score = computeNCC(image, template, x, y);

                if (score > bestScore) {
                    bestScore = score;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        return new TemplateMatchResult(bestScore, bestX, bestY);
    }

    private static double computeNCC(ImageProcessor img, ImageProcessor tmp, int offsetX, int offsetY) {
        int w = tmp.getWidth();
        int h = tmp.getHeight();

        double sumImg = 0, sumTmp = 0;
        double sumImgSq = 0, sumTmpSq = 0;
        double sumProduct = 0;
        int count = w * h;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double imgVal = img.getPixelValue(offsetX + x, offsetY + y);
                double tmpVal = tmp.getPixelValue(x, y);

                sumImg += imgVal;
                sumTmp += tmpVal;
                sumImgSq += imgVal * imgVal;
                sumTmpSq += tmpVal * tmpVal;
                sumProduct += imgVal * tmpVal;
            }
        }

        double meanImg = sumImg / count;
        double meanTmp = sumTmp / count;

        double numerator = sumProduct - count * meanImg * meanTmp;
        double denomImg = Math.sqrt(sumImgSq - count * meanImg * meanImg);
        double denomTmp = Math.sqrt(sumTmpSq - count * meanTmp * meanTmp);

        if (denomImg == 0 || denomTmp == 0) {
            return 0;
        }

        return numerator / (denomImg * denomTmp);
    }
}

