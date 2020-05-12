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

import struct.types.SLFeatureVector;

/**
 * Hand written Instance.
 *
 * @version 08/15/2006
 */
public class HandWritInstance extends SequenceInstance {
	private static Logger logger = Logger.getLogger(HandWritInstance.class.getName());

	public HandWritInstance(String[] sentence, String[] pos, String[] tags, SLFeatureVector fv,
			LinkedList[] predicates) {

		super(sentence, pos, tags, fv, predicates);
	}

	public HandWritInstance(String[] sentence, String[] tags, SLFeatureVector fv, LinkedList[] predicates) {
		super(sentence, tags, fv, predicates);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * struct.sequence.SequenceInstance#getFeatures(java.io.ObjectInputStream)
	 */
	@Override
	public SequenceFeatures getFeatures(ObjectInputStream in) throws IOException {

		String[] toks = input.sentence;

		SLFeatureVector fvs_ij[][][] = new SLFeatureVector[toks.length][tagAlphabet.size()][tagAlphabet.size()];
		SLFeatureVector fvs_j[][] = new SLFeatureVector[toks.length][tagAlphabet.size()];

		for (int k = 0; k < toks.length; k++) {
			for (int i = 0; i < tagAlphabet.size(); i++) {
				for (int j = 0; j < tagAlphabet.size(); j++) {
					SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);

					int indx = in.readInt();
					while (indx != -1) {
						double val = in.readDouble();
						prodFV = new SLFeatureVector(indx, val, prodFV);
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
					double val = in.readDouble();
					prodFV = new SLFeatureVector(indx, val, prodFV);
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
}
