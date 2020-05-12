/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.logging.Logger;

import cc.mallet.types.Alphabet;
import struct.alg.Predictor;
import struct.sequence.inference.KBestSequence;
import struct.sequence.inference.SequenceItem;
import struct.types.Features;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;

/**
 * A predictor for sequences.
 *
 * @version 07/15/2006
 */
public class SequencePredictor extends Predictor {
	private static final long serialVersionUID = 1L;

	private static Logger logger = Logger.getLogger(SequencePredictor.class.getName());

	// private final boolean disalowIntoStart;

	/**
	 *
	 * @param dimensions
	 *            The number of features
	 */
	public SequencePredictor(int dimensions) {
		super(dimensions);
		// disalowIntoStart = false;
	}

	// /**
	// *
	// * @param dimensions
	// * The number of features
	// */
	// public SequencePredictor(int dimensions, boolean disalowIntoStart) {
	// super(dimensions);
	// this.disalowIntoStart = disalowIntoStart;
	// }

	public void saveModel(String file) throws Exception {
		logger.info("Saving model ... ");
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
		out.writeObject(weights);
		out.writeObject(SequenceInstance.getDataAlphabet());
		out.writeObject(SequenceInstance.getTagAlphabet());
		out.close();
		logger.info("done.");
	}

	public void loadModel(String file) throws Exception {
		logger.info("Loading model ... ");
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
		weights = (double[]) in.readObject();
		SequenceInstance.setDataAlphabet((Alphabet) in.readObject());
		SequenceInstance.setTagAlphabet((Alphabet) in.readObject());
		in.close();
		logger.info("done.");
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see struct.alg.Predictor#decode(struct.types.Instance,
	 * struct.types.Features)
	 */
	@Override
	public SequencePrediction decode(SLInstance inst, Features feats) {
		return decode(inst, feats, 1);
	}

	/*
	 * (non-Javadoc)
	 *
	 * Inference - searches for the best tagging for a given instance.
	 * Iteratively considering tuples of tokens and searches for the highest
	 * scored label.
	 *
	 * @see struct.alg.Predictor#decode(struct.types.Instance,
	 * struct.types.Features, int)
	 */
	@Override
	public SequencePrediction decode(SLInstance inst, Features feats, int K) {
		SequenceInstance sinst = (SequenceInstance) inst;

		Alphabet tagAlphabet = SequenceInstance.getTagAlphabet();

		int instLength = sinst.getInput().sentence.length;

		KBestSequence bs = new KBestSequence(tagAlphabet, instLength, K);

		SequenceFeatures sf = (SequenceFeatures) feats;

		int startLab = tagAlphabet.lookupIndex("O");
		// System.out.println("startLab="+startLab);

		// init for the first token as no previous tokens exist
		bs.add(0, startLab, 0.0, new SLFeatureVector(-1, -1.0, null), null, 0);
		for (int i = 0; i < tagAlphabet.size(); i++)
			bs.add(0, i, Double.NEGATIVE_INFINITY, new SLFeatureVector(-1, -1.0, null), null, 0);

		// starting from the second token
		for (int n = 1; n < instLength; n++) {
			for (int e = 0; e < tagAlphabet.size(); e++) {
				for (int prev_e = 0; prev_e < tagAlphabet.size(); prev_e++) {
					// double addV = /*disalowIntoStart && e == startLab ?
					// Double.NEGATIVE_INFINITY : */0.0;
					SLFeatureVector fv_ij = sf.getFeatureVector(n, prev_e, e);
					double prob_ij = score(fv_ij);
					SLFeatureVector fv_i = sf.getFeatureVector(n, e);
					double prob_i = score(fv_i);
					SequenceItem[] items = bs.getItems(n - 1, prev_e);

					if (items == null)
						continue;

					int strt = 0;
					for (int i = 0; i < items.length; i++) {
						if (items[i].prob == Double.NEGATIVE_INFINITY)
							continue;
						strt = bs.add(n, e, prob_ij + prob_i + items[i].prob /* + addV */,
								SLFeatureVector.cat(fv_ij, fv_i), items[i], strt);
						if (strt < 0)
							break;
					}
				}
			}
		}
		return bs.getBestSequences();
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see struct.alg.Predictor#grow(int)
	 */
	@Override
	public void grow(int newSize) {
		double[] newWeights = new double[newSize];
		double[] newAvg_weights = new double[newSize];
		for (int i = 0; i < weights.length; i++) {
			newWeights[i] = weights[i];
			newAvg_weights[i] = avg_weights[i];
		}
		weights = newWeights;
		avg_weights = newAvg_weights;
	}
}
