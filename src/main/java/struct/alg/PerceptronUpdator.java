/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.alg;

import struct.types.Features;
import struct.types.Prediction;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;
import struct.types.SLLabel;

/**
 * The perceptron updator.
 *
 * @version 08/15/2006
 */
public class PerceptronUpdator implements OnlineUpdator {

	/*
	 * (non-Javadoc)
	 *
	 * @see struct.alg.OnlineUpdator#update(struct.types.Instance,
	 * struct.types.Features, struct.alg.Predictor, double)
	 */
	@Override
	public void update(SLInstance inst, Features feats, Predictor predictor, double avg_upd) {

		Prediction pred = predictor.decode(inst, feats); // inference
		SLLabel label = pred.getBestLabel();
		SLFeatureVector guessed_fv = label.getFeatureVectorRepresentation(); // phi(x,y_pred)
		SLFeatureVector corr_fv = inst.getLabel().getFeatureVectorRepresentation(); // phi(x,y_gold)

		double[] weights = predictor.weights;
		double[] avg_weights = predictor.avg_weights;

		// note that for features that are corrected no change is made to w
		// for mistakes we boost features for y_gold and reduce features for
		// y_pred

		// w = w + \phi(x,y_gold)
		for (SLFeatureVector curr = corr_fv; curr.next != null; curr = curr.next) {
			if (curr.index >= 0) {
				weights[curr.index] += curr.value;
				avg_weights[curr.index] += avg_upd * curr.value;
			}
		}
		// w = w - \phi(x,y_pred)
		for (SLFeatureVector curr = guessed_fv; curr.next != null; curr = curr.next) {
			if (curr.index >= 0) {
				weights[curr.index] -= curr.value;
				avg_weights[curr.index] -= avg_upd * curr.value;
			}
		}
	}
}
