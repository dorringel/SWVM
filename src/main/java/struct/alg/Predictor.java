/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.alg;

import java.io.Serializable;

import struct.types.Features;
import struct.types.Prediction;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;

/**
 * The predictor which predicts targets of instances.
 *
 * @version 08/15/2006
 */
public abstract class Predictor implements Serializable {

	public double[] weights;
	public double[] avg_weights;

	/**
	 * @param dimensions
	 *            - the number of features
	 */
	public Predictor(int dimensions) {
		weights = new double[dimensions];
		avg_weights = new double[dimensions];
		for (int i = 0; i < dimensions; i++) {
			weights[i] = 0.0;
			avg_weights[i] = 0.0;
		}
	}

	/**
	 * Averages the weights.
	 */
	public void averageWeights(int factor) {
		double fact = factor;
		for (int i = 0; i < avg_weights.length; i++)
			weights[i] = avg_weights[i] / fact;
	}

	/**
	 * Computes the score of the SLFeatureVector
	 */
	public double score(SLFeatureVector fv) {
		double score = 0.0;
		for (SLFeatureVector curr = fv; curr != null; curr = curr.next) {
			if (curr.index >= 0)
				score += weights[curr.index] * curr.value;
		}
		return score;
	}

	/**
	 * Predicts for an instance and its features.
	 */
	public abstract Prediction decode(SLInstance inst, Features feats);

	/**
	 * Predicts for an instance and its features based on K-best.
	 */
	public abstract Prediction decode(SLInstance inst, Features feats, int K);

	/**
	 * Grows this predictor to make it ready for training with additional
	 * features.
	 *
	 * @param newSize
	 *            - the new number of features
	 */
	public abstract void grow(int newSize);
}
