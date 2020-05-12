/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.logging.Logger;

import cc.mallet.fst.Transducer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.ArrayListSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.Sequence;
import struct.alg.OnlineLearner;
import struct.alg.OnlineUpdator;
import struct.alg.Predictor;
import struct.types.Features;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;

/**
 * A linear model for sequences.
 *
 * @author Surya Prakash Bachoti
 * @version 07/15/2006
 */
public class LinearSequenceModel extends Transducer {
	private static Logger logger = Logger.getLogger(LinearSequenceModel.class.getName());

	private Predictor predictor;
	// The alphabet containing labels
	private Alphabet tagAlphabet;
	// The alphabet containing features
	private Alphabet dataAlphabet;
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor for objects of class LinearSequenceModel.
	 *
	 * @param numFeats
	 *            - the number of features
	 * @param tagAlphabet
	 *            - the label alphabet
	 * @param dataAlphabet
	 *            - the feature alphabet
	 */
	public LinearSequenceModel(int numFeats, Alphabet tagAlphabet, Alphabet dataAlphabet) {
		predictor = new SequencePredictor(numFeats);
		this.tagAlphabet = tagAlphabet;
		this.dataAlphabet = dataAlphabet;
	}

	public LinearSequenceModel(InstanceListDataManager manager) {
		this(manager.dataAlphabet.size(), manager.tagAlphabet, manager.dataAlphabet);
	}

	@Override
	public int numStates() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public State getState(int index) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Iterator initialStateIterator() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * Trains the model.
	 *
	 * @param train_iList
	 *            - the training InstanceList
	 * @param numIter
	 *            - the number of iterations
	 * @param updator
	 *            - the OnlineUpdator
	 * @throws IOException
	 */
	public void train(InstanceList train_iList, int numIter, OnlineUpdator updator) throws IOException {
		SequenceInstance[] trainingData = convert(train_iList);
		OnlineLearner onlinelearner = new OnlineLearner();

		onlinelearner.train(trainingData, updator, predictor, numIter);
	}

	/**
	 * Converts the given sequence into another sequence according to this
	 * transducer.
	 *
	 * @param input
	 *            - Input sequence
	 * @return Sequence - output by this transudcer
	 */
	@Override
	@SuppressWarnings("unchecked")
	public Sequence transduce(Sequence input) {
		ArrayListSequence seq = new ArrayListSequence();
		if (!(input instanceof FeatureVectorSequence)) {
			// throw new BadInstanceException(); //unsupported
			return null;
		}

		SLInstance inst = getInstance((FeatureVectorSequence) input);

		Features feats = inst.getFeatures();

		SequencePrediction prediction = (SequencePrediction) predictor.decode(inst, feats);

		String[] pred_spans = prediction.getBestLabel().tags;

		// first entry is a dummy "O" tag.
		for (int i = 1; i < pred_spans.length; i++) {
			seq.add(pred_spans[i]);
		}

		return seq;
	}

	/**
	 * Converts a FeatureVectorSequence into a SequenceInstance.
	 */
	private SequenceInstance getInstance(FeatureVectorSequence fvs) {
		LinkedList[] predicates = getPredicates(fvs);

		String[] tags = new String[fvs.size() + 1];

		SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null));
		fv.sort();

		return new SequenceInstance(new String[tags.length], tags, fv, predicates);
	}

	/**
	 * Converts the InstanceList into an array of SequenceInstances
	 *
	 * @throws IOException
	 */
	private SequenceInstance[] convert(InstanceList iList) throws IOException {

		LinkedList lt = new LinkedList();

		logger.info("Creating feature vectors and/or forests ...");

		Instance inst;
		FeatureVectorSequence fvs;
		LabelSequence ls;
		for (int i = 0; i < iList.size(); i++) {

			inst = iList.get(i);
			fvs = (FeatureVectorSequence) inst.getData();
			ls = (LabelSequence) inst.getTarget();

			String[] tags = new String[ls.size() + 1];
			tags[0] = "O";
			for (int j = 0; j < ls.size(); j++)
				tags[j + 1] = (String) ls.get(j);

			LinkedList[] predicates = getPredicates(fvs);

			SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null));
			fv.sort();

			SequenceInstance si = new SequenceInstance(new String[tags.length], tags, fv, predicates);

			lt.add(si);
		}

		SequenceInstance[] si = new SequenceInstance[lt.size()];
		for (int i = 0; i < si.length; i++) {
			si[i] = (SequenceInstance) lt.get(i);
		}

		return si;
	}

	private LinkedList[] getPredicates(FeatureVectorSequence fvs) {
		LinkedList[] predicates = new LinkedList[fvs.size() + 1];

		predicates[0] = new LinkedList();

		for (int i = 0; i < fvs.size(); i++) {
			predicates[i + 1] = new LinkedList();
			FeatureVector fv = fvs.getFeatureVector(i);
			int[] nonZeroIndices = fv.getIndices();
			String[] list = new String[nonZeroIndices.length];
			for (int j = 0; j < list.length; j++) {
				list[j] = "feat" + nonZeroIndices[j];
				predicates[i + 1].add(list[j]);
			}
		}
		return predicates;
	}

	private SLFeatureVector createFeatureVector(LinkedList predicates, String prev, String next, SLFeatureVector fv) {

		String s1 = prev + "_" + next;
		fv = fv.add("TRANS=" + s1, 1.0, dataAlphabet);

		return fv;
	}

	private SLFeatureVector createFeatureVector(LinkedList predicates, String next, SLFeatureVector fv) {

		String s2 = next;
		for (int j = 0; j < predicates.size(); j++) {
			String pred = (String) predicates.get(j);
			fv = fv.add(s2 + "_" + pred, 1.0, dataAlphabet);
		}

		return fv;
	}

	private SLFeatureVector createFeatureVector(LinkedList[] predicates, String[] tags, SLFeatureVector fv) {

		for (int i = 1; i < predicates.length; i++) {
			fv = createFeatureVector(predicates[i], (i == 0 ? "O" : tags[i - 1]), tags[i], fv);
			fv = createFeatureVector(predicates[i], tags[i], fv);
		}

		return fv;
	}

	/**
	 * Grows the model to prepare for training with new features.
	 *
	 * @param dataAlphabet
	 *            - the feature alphabet
	 * @param tagAlphabet
	 *            - the label alphabet
	 */
	public void grow(Alphabet dataAlphabet, Alphabet tagAlphabet) {
		if (this.dataAlphabet != dataAlphabet || this.tagAlphabet != tagAlphabet) {
			throw new IllegalArgumentException("A different dataAlphabet or tagAlphabet. Not allowed!");
		}
		predictor.grow(this.dataAlphabet.size());
	}

	private void writeObject(ObjectOutputStream stream) throws IOException {
		stream.writeObject(dataAlphabet);
		stream.writeObject(tagAlphabet);
		stream.writeObject(predictor);
	}

	private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
		dataAlphabet = (Alphabet) stream.readObject();
		tagAlphabet = (Alphabet) stream.readObject();
		predictor = (Predictor) stream.readObject();
		SequenceInstance.setDataAlphabet(dataAlphabet);
		SequenceInstance.setTagAlphabet(tagAlphabet);
	}
}
