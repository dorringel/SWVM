/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.classification;

import java.io.IOException;
import java.util.LinkedList;
import java.util.logging.Logger;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Label;
import struct.alg.KBestMiraUpdator;
import struct.alg.OnlineLearner;

/**
 * The KBestMira classifier trainer which trains a linear classifier.
 *
 * @version 08/22/2006
 */
public class KBestMiraClassifierTrainer extends ClassifierTrainer {
	private static Logger logger = Logger.getLogger(KBestMiraClassifierTrainer.class.getName());
	private final int K;
	private double C = Double.POSITIVE_INFINITY;
	private final int iterations;
	private boolean averaging = true;

	/**
	 * Trains default # of iterations (5). Trains without slack variable (C =
	 * infinity). Trains with averaging.
	 *
	 * @param K
	 *            - the K in KBestMira
	 */
	public KBestMiraClassifierTrainer(int K) {
		this(K, 5);
	}

	/**
	 * Trains without slack variable (C). Trains with averaging.
	 *
	 * @param K
	 *            - the K in KBestMira
	 * @param iterations
	 *            - the number of iterations to use in training
	 */
	public KBestMiraClassifierTrainer(int K, int iterations) {
		this(K, iterations, Double.POSITIVE_INFINITY);
	}

	/**
	 * Trains with averaging.
	 *
	 * @param K
	 *            - the K in KBestMira
	 * @param iterations
	 *            - the number of iterations to use in training
	 * @param C
	 *            - The slack variable (clipping). Reasonable values are .1,
	 *            .01. 1 usually ensures a full update, 0 will never update.
	 */
	public KBestMiraClassifierTrainer(int K, int iterations, double C) {
		this(K, iterations, C, true);
	}

	/**
	 * @param averaging
	 *            - Train with averaging.
	 * @param K
	 *            - the K in KBestMira
	 * @param iterations
	 *            - the number of iterations to use in training
	 * @param C
	 *            - The slack variable (clipping). Reasonable values are .1,
	 *            .01. 1 usually ensures a full update, 0 will never update.
	 */
	public KBestMiraClassifierTrainer(int K, int iterations, double C, boolean averaging) {
		this.K = K;
		this.iterations = iterations;
		this.C = C;
		this.averaging = averaging;
	}

	// TODO: Elad removed
	// /**
	// * Creates a new linear classifier and trains it or else just trains the
	// * classifier if the classifier is not null.
	// */
	// @Override
	// public Classifier train(InstanceList trainingSet, InstanceList
	// validationSet, InstanceList testSet,
	// ClassifierEvaluating evaluator, Classifier initialClassifier) {
	//
	// if (trainingSet == null)
	// throw new IllegalArgumentException("Training set cannot be null!");
	//
	// if (validationSet != null) {
	// if (!(validationSet.getDataAlphabet() == trainingSet.getDataAlphabet()
	// && validationSet.getTargetAlphabet() == trainingSet.getTargetAlphabet()))
	// {
	// throw new IllegalArgumentException(
	// "Validation set alphabet does not match training set alphabet,
	// aborting!");
	// }
	// }
	//
	// if (testSet != null) {
	// if (!(testSet.getDataAlphabet() == trainingSet.getDataAlphabet()
	// && testSet.getTargetAlphabet() == trainingSet.getTargetAlphabet())) {
	// throw new IllegalArgumentException("Test set alphabet does not match
	// training set alphabet, aborting!");
	// }
	// }
	//
	// if (initialClassifier != null) {
	// if (!(initialClassifier.getAlphabet() == trainingSet.getDataAlphabet()
	// && initialClassifier.getLabelAlphabet() ==
	// trainingSet.getTargetAlphabet())) {
	// throw new IllegalArgumentException(
	// "Initial classifier alphabet does not match training set alphabet,
	// aborting!");
	// }
	// return modify(trainingSet, initialClassifier);
	// } else {
	// return getNewClassifier(trainingSet);
	// }
	// }

	private Classifier getNewClassifier(InstanceList trainingSet) {
		LinearClassifier linearClassifier = new LinearClassifier(trainingSet.getPipe());
		linearClassifier.createPhiAlphabet(trainingSet);
		try {
			trainClassifier(linearClassifier, trainingSet, this.iterations);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return linearClassifier;
	}

	/**
	 * Trains the classifier.
	 *
	 * @param numIter
	 *            - the number of iterations
	 * @throws IOException
	 */
	private void trainClassifier(LinearClassifier linearClassifier, InstanceList trainingSet, int numIter)
			throws IOException {
		linearClassifier.setInstanceAlphabets();
		ClassificationInstance[] trainingData = readData(trainingSet);
		OnlineLearner onlinelearner = new OnlineLearner(this.averaging);

		onlinelearner.train(trainingData, new KBestMiraUpdator(this.K, this.C), linearClassifier.getPredictor(),
				numIter);

	}

	private Classifier modify(InstanceList trainingSet, Classifier initialClassifier) {
		LinearClassifier linearClassifier;
		try {
			linearClassifier = (LinearClassifier) initialClassifier;
		} catch (ClassCastException e) {
			throw new IllegalArgumentException("The classifier must be a linear classifier");
		}
		linearClassifier.grow(trainingSet.getTargetAlphabet(), trainingSet);
		try {
			trainClassifier(linearClassifier, trainingSet, this.iterations);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return linearClassifier;
	}

	private ClassificationInstance[] readData(InstanceList iList) throws IOException {

		LinkedList lt = new LinkedList();

		logger.info("Creating feature vectors and/or forests ...");

		Instance inst;
		FeatureVector fv;
		Label label;
		for (int i = 0; i < iList.size(); i++) {
			inst = iList.get(i);
			fv = (FeatureVector) inst.getData();
			label = (Label) inst.getTarget();

			String tag = (String) label.getEntry();

			ClassificationInstance si = new ClassificationInstance(tag, fv);

			lt.add(si);
		}
		ClassificationInstance[] si = new ClassificationInstance[lt.size()];
		for (int i = 0; i < si.length; i++) {
			si[i] = (ClassificationInstance) lt.get(i);
		}

		return si;
	}

	@Override
	public Classifier getClassifier() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Classifier train(InstanceList trainingSet) {
		if (trainingSet == null)
			throw new IllegalArgumentException("Training set cannot be null!");
		if (validationSet != null) {
			if (!(validationSet.getDataAlphabet() == trainingSet.getDataAlphabet()
					&& validationSet.getTargetAlphabet() == trainingSet.getTargetAlphabet())) {
				throw new IllegalArgumentException(
						"Validation set alphabet does not match training set alphabet, aborting!");
			}
		}

		return getNewClassifier(trainingSet);
	}
}
