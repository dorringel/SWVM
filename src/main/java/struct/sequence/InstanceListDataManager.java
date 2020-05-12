/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.LinkedList;

import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import struct.types.SLFeatureVector;

/**
 * A DataManager to manage Mallet Instances.
 *
 * @version 08/15/2006
 */
public class InstanceListDataManager extends SequenceDataManager {
	// The managed instanceList
	private final InstanceList iList;
	private HashMap<String, Boolean> featureMap = null;

	/**
	 * @param iList
	 *            The Mallet InstanceList
	 */
	public InstanceListDataManager(InstanceList iList) {
		this.iList = iList;
		tagAlphabet = iList.getTargetAlphabet();
	}

	public InstanceListDataManager(InstanceList iList, HashMap<String, Boolean> featureMap) {
		this.iList = iList;
		tagAlphabet = iList.getTargetAlphabet();
		this.featureMap = featureMap;
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see
	 * struct.sequence.SequenceDataManager#getPredicates(java.lang.String[],
	 * java.lang.String[])
	 */
	@Override
	public LinkedList[] getPredicates(String[] toks, String[] pos) {
		return null;
	}

	/**
	 * Creates the predicates from a Mallet FeatureVectorSequence.
	 */
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

	/**
	 * Creates alphabets.
	 *
	 * @param createUnsupported
	 *            whether to add unsupported features. This means whether to
	 *            create a feature for some label if we have only ever seen it
	 *            with another label. Adding these to the alphabet allows us to
	 *            learn a weight for them. Adding unsupported features has been
	 *            shown to help with CRF training.
	 * @throws IOException
	 */
	public void createAlphabets(boolean createUnsupported) throws IOException {
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
			createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null), this.featureMap);

			if (createUnsupported) {
				createU(predicates);
			}
		}
	}

	/**
	 * Grows the data alphabet with new features.
	 */
	public void grow(InstanceList iList, boolean createUnsupported) {
		if (tagAlphabet != iList.getTargetAlphabet()) {
			throw new IllegalArgumentException("Cannot use a different alphabet");
		}

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
			SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null), this.featureMap);

			if (createUnsupported)
				createU(predicates);
		}
	}

	/**
	 * Sets the SequenceInstance's alphabets.
	 */
	public void setInstanceAlphabets() {
		SequenceInstance.setTagAlphabet(tagAlphabet);
		SequenceInstance.setDataAlphabet(dataAlphabet);
	}

	/**
	 * Reads instances from a Mallet InstanceList and converts them into
	 * SequenceInstances.
	 *
	 * @throws IOException
	 */
	public SequenceInstance[] readData(InstanceList iList) throws IOException {

		LinkedList lt = new LinkedList();

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
			SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null), this.featureMap);
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

	/*
	 * (non-Javadoc)
	 *
	 * @see struct.sequence.SequenceDataManager#createForest(struct.sequence.
	 * SequenceInstance, java.util.LinkedList[], java.io.ObjectOutputStream)
	 */
	@Override
	public void createForest(SequenceInstance inst, LinkedList[] predicates, ObjectOutputStream out) {
		String[] toks = inst.getInput().sentence;

		try {

			for (int k = 0; k < predicates.length; k++) {
				for (int i = 0; i < tagAlphabet.size(); i++) {
					for (int j = 0; j < tagAlphabet.size(); j++) {
						SLFeatureVector nfv = createFeatureVector(predicates[k], (String) tagAlphabet.lookupObject(i),
								(String) tagAlphabet.lookupObject(j), new SLFeatureVector(-1, -1.0, null));

						nfv.sort();

						for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
							if (curr.index >= 0)
								out.writeInt(curr.index);
						out.writeInt(-1);
					}
				}
			}

			out.writeInt(-2);

			for (int k = 0; k < predicates.length; k++) {
				for (int j = 0; j < tagAlphabet.size(); j++) {
					SLFeatureVector nfv = createFeatureVector(predicates[k], (String) tagAlphabet.lookupObject(j),
							new SLFeatureVector(-1, -1.0, null));

					nfv.sort();

					for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
						if (curr.index >= 0)
							out.writeInt(curr.index);
					out.writeInt(-1);
				}
			}
			out.writeInt(-3);
		} catch (IOException e) {
		}
	}
}
