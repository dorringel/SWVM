/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.classification;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.logging.Logger;

import cc.mallet.types.Alphabet;
import cc.mallet.types.LabelAlphabet;
import struct.alg.Predictor;
import struct.types.Features;
import struct.types.Prediction;
import struct.types.SLInstance;

/**
 * The predictor for classification.
 *
 * @version 08/22/2006
 */
public class ClassificationPredictor extends Predictor {
	private static Logger logger = Logger.getLogger(ClassificationPredictor.class.getName());
	private static final long serialVersionUID = -2771793270662248278L;

	/**
	 * @param dimensions
	 *            - the total number of features.
	 */
	public ClassificationPredictor(int dimensions) {
		super(dimensions);
	}

	/**
	 * Evaluates the prediction for the Features
	 */
	@Override
	public Prediction decode(SLInstance inst, Features feats) {
		int K = ClassificationInstance.getTagAlphabet().size();
		return decode(inst, feats, K);
	}

	/**
	 * Evaluates the prediction for the Features
	 */
	@Override
	public ClassificationPrediction decode(SLInstance inst, Features feats, int K) {
		Features sf = feats;

		return getTopKLabels(sf, K);
	}

	private ClassificationPrediction getTopKLabels(Features sf, int K) {
		Alphabet tagAlphabet = ClassificationInstance.getTagAlphabet();

		int[] indices = new int[tagAlphabet.size()];
		for (int i = 0; i < indices.length; i++)
			indices[i] = i;

		double[] scores = new double[tagAlphabet.size()];
		for (int i = 0; i < scores.length; i++)
			scores[i] = score(((ClassificationFeatures) sf).getFeatureVector(i));

		sort(indices, scores);

		ClassificationLabel[] labels = new ClassificationLabel[K];

		int min = Math.min(K, tagAlphabet.size());
		for (int i = 0; i < min; i++)
			labels[i] = new ClassificationLabel((String) (tagAlphabet.lookupObject(indices[i])),
					((ClassificationFeatures) sf).getFeatureVector(indices[i]));

		return new ClassificationPrediction(labels, scores);
	}

	private void sort(int[] indices, double[] scores) {
		for (int i = 0; i < indices.length - 1; i++) {
			for (int j = 0; j < indices.length - 1 - i; j++) {
				if (scores[j + 1] > scores[j]) { /* compare the two neighbors */
					/* swap */
					double tmpScore = scores[j];
					scores[j] = scores[j + 1];
					scores[j + 1] = tmpScore;

					int tmpIndex = indices[j];
					indices[j] = indices[j + 1];
					indices[j + 1] = tmpIndex;
				}
			}
		}
	}

	/**
	 * Increases the size of the predictor's weights to make the predictor ready
	 * for training with additional features.
	 */
	@Override
	public void grow(int newSize) {
		int old_size = weights.length;
		if (old_size == newSize)
			return;
		double[] newWeights = new double[newSize];
		double[] newAvg_weights = new double[newSize];
		for (int i = 0; i < weights.length; i++) {
			newWeights[i] = weights[i];
			newAvg_weights[i] = avg_weights[i];
		}
		weights = newWeights;
		avg_weights = newAvg_weights;
	}

	/**
	 * Saves the model in the file
	 */
	protected void saveModel(String file) throws Exception {
		logger.info("Saving model ... ");
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
		out.writeObject(weights);
		out.writeObject(ClassificationInstance.getDataAlphabet());
		out.writeObject(ClassificationInstance.getTagAlphabet());
		out.close();
		logger.info("done.");
	}

	/**
	 * Loads the model from the file.
	 */
	protected void loadModel(String file) throws Exception {
		logger.info("Loading model ... ");
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
		weights = (double[]) in.readObject();
		ClassificationInstance.setDataAlphabet((Alphabet) in.readObject());
		ClassificationInstance.setTagAlphabet((LabelAlphabet) in.readObject());
		in.close();
		logger.info("done.");
	}
}
