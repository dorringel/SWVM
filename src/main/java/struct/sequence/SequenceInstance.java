/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.LinkedList;
import java.util.logging.Logger;

import cc.mallet.types.Alphabet;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;

/**
 * A sequence instance.
 *
 * @version 07/15/2006
 */
public class SequenceInstance implements SLInstance {
	private static Logger logger = Logger.getLogger(SequenceInstance.class.getName());

	protected LinkedList[] predicates;
	protected SequenceInput input;
	protected SequenceLabel label;

	protected static Alphabet tagAlphabet;
	protected static Alphabet dataAlphabet;

	public SequenceInstance() {
	}

	public SequenceInstance(String[] sentence, String[] pos, String[] tags, SLFeatureVector fv,
			LinkedList[] predicates) {

		input = new SequenceInput(sentence, pos);

		label = new SequenceLabel(tags, fv);

		this.predicates = predicates;

	}

	public SequenceInstance(String[] sentence, String[] tags, SLFeatureVector fv, LinkedList[] predicates) {

		input = new SequenceInput(sentence);

		label = new SequenceLabel(tags, fv);

		this.predicates = predicates;

	}

	public SequenceInput getInput() {
		return input;
	}

	public SequenceLabel getLabel() {
		return label;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.Instance#getFeatures()
	 */
	public SequenceFeatures getFeatures() {
		String[] toks = input.sentence;

		SLFeatureVector fvs_ij[][][] = new SLFeatureVector[toks.length][tagAlphabet.size()][tagAlphabet.size()];
		SLFeatureVector fvs_j[][] = new SLFeatureVector[toks.length][tagAlphabet.size()];

		for (int k = 0; k < predicates.length; k++) {
			for (int i = 0; i < tagAlphabet.size(); i++) {
				for (int j = 0; j < tagAlphabet.size(); j++) {
					SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);

					SLFeatureVector nfv = createFeatureVector(predicates[k], (String) tagAlphabet.lookupObject(i),
							(String) tagAlphabet.lookupObject(j), new SLFeatureVector(-1, -1.0, null));

					nfv.sort();

					for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
						if (curr.index >= 0)
							prodFV = new SLFeatureVector(curr.index, 1.0, prodFV);

					fvs_ij[k][i][j] = prodFV;
				}
			}
		}

		for (int k = 0; k < toks.length; k++) {
			for (int i = 0; i < tagAlphabet.size(); i++) {
				SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);

				SLFeatureVector nfv = createFeatureVector(predicates[k], (String) tagAlphabet.lookupObject(i),
						new SLFeatureVector(-1, -1.0, null));

				nfv.sort();

				for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
					if (curr.index >= 0)
						prodFV = new SLFeatureVector(curr.index, 1.0, prodFV);

				fvs_j[k][i] = prodFV;

			}
		}

		return new SequenceFeatures(fvs_ij, fvs_j);

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

	public SequenceFeatures getFeatures(ObjectInputStream in) throws IOException {

		String[] toks = input.sentence;

		SLFeatureVector fvs_ij[][][] = new SLFeatureVector[toks.length][tagAlphabet.size()][tagAlphabet.size()];
		SLFeatureVector fvs_j[][] = new SLFeatureVector[toks.length][tagAlphabet.size()];

		for (int k = 0; k < toks.length; k++) {
			for (int i = 0; i < tagAlphabet.size(); i++) {
				for (int j = 0; j < tagAlphabet.size(); j++) {
					SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);
					int indx = in.readInt();
					// System.out.print(" count="+(count++)+" indx="+indx);
					while (indx != -1) {
						prodFV = new SLFeatureVector(indx, 1.0, prodFV);
						indx = in.readInt();
					}

					fvs_ij[k][i][j] = prodFV;

				}
			}
		}

		int i1 = in.readInt();
		if (i1 != -2) {
			logger.severe("Feature Vector I/O problem 1:" + i1);
			logger.severe(toks[1] + "\t" + tagAlphabet.size());
			throw new IOException("Feature Vector I/O problem 1:" + i1 + ", " + toks[1] + "\t" + tagAlphabet.size());
		}

		for (int k = 0; k < toks.length; k++) {
			for (int i = 0; i < tagAlphabet.size(); i++) {
				SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);

				int indx = in.readInt();
				while (indx != -1) {
					prodFV = new SLFeatureVector(indx, 1.0, prodFV);
					indx = in.readInt();
				}

				fvs_j[k][i] = prodFV;

			}
		}

		i1 = in.readInt();
		if (i1 != -3) {
			logger.severe("Feature Vector I/O problem 2:" + i1);
			logger.severe(toks[1] + "\t" + tagAlphabet.size() + "\t");
			throw new IOException("Feature Vector I/O problem 2:" + i1 + ", " + toks[1] + "\t" + tagAlphabet.size());

		}

		return new SequenceFeatures(fvs_ij, fvs_j);

	}

	public static void setTagAlphabet(Alphabet tagAlphabet1) {
		tagAlphabet = tagAlphabet1;
	}

	public static Alphabet getTagAlphabet() {
		return tagAlphabet;
	}

	public static void setDataAlphabet(Alphabet dataAlphabet1) {
		dataAlphabet = dataAlphabet1;
	}

	public static Alphabet getDataAlphabet() {
		return dataAlphabet;
	}
}
