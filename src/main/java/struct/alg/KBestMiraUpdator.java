/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.alg;

import struct.solver.QPSolver;
import struct.types.Features;
import struct.types.Prediction;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;
import struct.types.SLLabel;

/**
 * The KBestMiraUpdator updates the weights of the predictor.
 *
 * @version 08/15/2006
 */
public class KBestMiraUpdator implements OnlineUpdator {

	private double C = Double.POSITIVE_INFINITY;
	private final int K;

	/**
	 * @param K
	 *            - the K in KBestMira
	 */
	public KBestMiraUpdator(int K) {
		this.K = K;
	}

	/**
	 *
	 * @param K
	 *            - the K in KBestMira
	 * @param C
	 *            - The slack variable (clipping). Reasonable values are .1,
	 *            .01. 1 usually ensures a full update, 0 will never update.
	 */
	public KBestMiraUpdator(int K, double C) {
		this(K);
		this.C = C;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.alg.OnlineUpdator#update(struct.types.Instance,
	 * struct.types.Features, struct.alg.Predictor, double)
	 */
	public void update(SLInstance inst, Features feats, Predictor predictor, double avg_upd) {

		Prediction pred = predictor.decode(inst, feats, K);

		SLLabel[] labels = new SLLabel[K];
		int k_new = 0;
		for (int k = 0; k < K; k++) {
			labels[k] = pred.getLabelByRank(k);
			if (labels[k] == null)
				break;
			k_new = k + 1;
		}

		SLFeatureVector corr_fv = inst.getLabel().getFeatureVectorRepresentation();

		SLFeatureVector[] guessed_fvs = new SLFeatureVector[k_new];
		double[] b = new double[k_new];
		double[] lam_dist = new double[k_new];
		SLFeatureVector[] dist = new SLFeatureVector[k_new];

		for (int k = 0; k < k_new; k++) {
			guessed_fvs[k] = labels[k].getFeatureVectorRepresentation();

			// compute [scalar] hamming distance (number of locations in which correct tag and predicted tag are different (for example "O" <> "RNA")
			b[k] = inst.getLabel().loss(labels[k]);

			// compute [scalar] w.dot(phi(x,y)) - w.dot(phi(x,y')) = w.dot(phi(x,y) - phi(x,y'))
			lam_dist[k] = predictor.score(corr_fv) - predictor.score(guessed_fvs[k]);

			// b[k] = loss[k] - w.dot(phi)[k]
			b[k] -= lam_dist[k];

			// compute [vector] phi(x,y) - phi(x,y')[k]
 			dist[k] = SLFeatureVector.getDistVector(corr_fv, guessed_fvs[k]);
		}

		double[] alpha = null;
		if (this.C != Double.POSITIVE_INFINITY)
			alpha = QPSolver.hildreth(dist, b, this.C);
		else
			alpha = QPSolver.hildreth(dist, b);

		double[] weights = predictor.weights;
		double[] avg_weights = predictor.avg_weights;

		SLFeatureVector fv = null;
		for (int k = 0; k < k_new; k++) {
			fv = dist[k];
			for (SLFeatureVector curr = fv; curr != null; curr = curr.next) {
				if (curr.index < 0)
					continue;
				weights[curr.index] += alpha[k] * curr.value;
				avg_weights[curr.index] += avg_upd * alpha[k] * curr.value;
			}
		}

	}

}
