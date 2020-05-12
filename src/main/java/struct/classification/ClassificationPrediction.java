/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.classification;

import struct.types.Prediction;

/**
 * A set of predicted labels, in order.
 *
 * @version 08/20/2006
 */
public class ClassificationPrediction implements Prediction {
	//The predicted labels
	private ClassificationLabel[] labels;
	//The scores
	private double[] scores;
	
	public ClassificationPrediction(ClassificationLabel[] labels, double[] scores) {
		this.labels = labels;
		this.scores = scores;
	}
	
	public ClassificationLabel getBestLabel() { 
		return labels[0]; 
	}
	
	public ClassificationLabel getLabelByRank(int rank) {
		if(rank >= labels.length)
			return null;
		return labels[rank];
	}
	
	public int getNumLabels() { 
		return labels.length; 
	}
	
	public double[] getScores() {
		return scores;
	}
}
