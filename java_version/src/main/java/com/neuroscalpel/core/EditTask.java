package com.neuroscalpel.core;

/** Represents a single hallucination-correction task parsed from Phase 1. */
public class EditTask {
    private final int index;
    private String analysis;
    private String trickPrompt;
    private String subject;
    private String wrongValue;
    private String correctValue;
    private int    targetLayer = -1;
    private int    targetPoint = -1;

    public EditTask(int index, String analysis, String trickPrompt,
                    String subject, String wrongValue, String correctValue) {
        this.index        = index;
        this.analysis     = analysis;
        this.trickPrompt  = trickPrompt;
        this.subject      = subject;
        this.wrongValue   = wrongValue;
        this.correctValue = correctValue;
    }

    public int    getIndex()        { return index; }
    public String getAnalysis()     { return analysis; }
    public String getTrickPrompt()  { return trickPrompt; }
    public String getSubject()      { return subject; }
    public String getWrongValue()   { return wrongValue; }
    public String getCorrectValue() { return correctValue; }
    public int    getTargetLayer()  { return targetLayer; }
    public int    getTargetPoint()  { return targetPoint; }

    public void setTargetLayer(int l) { targetLayer = l; }
    public void setTargetPoint(int p) { targetPoint = p; }

    @Override
    public String toString() {
        return String.format("[%d] %s | %s → %s", index+1, subject, wrongValue, correctValue);
    }
}
