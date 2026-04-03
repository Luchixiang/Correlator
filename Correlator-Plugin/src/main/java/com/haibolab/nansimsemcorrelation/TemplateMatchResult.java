package com.haibolab.nansimsemcorrelation;

class TemplateMatchResult {
    double score;
    int x, y;

    TemplateMatchResult(double score, int x, int y) {
        this.score = score;
        this.x = x;
        this.y = y;
    }
}
