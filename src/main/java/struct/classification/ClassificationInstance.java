/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.classification;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.LabelAlphabet;
import struct.types.Features;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;
import struct.types.SLLabel;

/**
 * Internal representation of a Mallet Instance for the classification.
 *
 * @version 08/18/2006
 */
public class ClassificationInstance implements SLInstance {

	// The featureVector of the Mallet Instance
	protected FeatureVector featureVector;
	// The target classification label for this instance
	protected ClassificationLabel label;
	// The labelAlphabet containing all labels
	protected static LabelAlphabet tagAlphabet;
	// The alphabet containing all features
	protected static Alphabet dataAlphabet;

	/**
	 * @param tag
	 *            - The target label for this instance.
	 * @param featureVector
	 *            - The Mallet FeatureVector of this instance.
	 */
	public ClassificationInstance(String tag, FeatureVector featureVector) {
		SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);

		SLFeatureVector nfv = createFeatureVector(featureVector, tag, new SLFeatureVector(-1, -1.0, null));
		nfv.sort();

		for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
			if (curr.index >= 0)
				prodFV = new SLFeatureVector(curr.index, 1.0, prodFV);

		label = new ClassificationLabel(tag, prodFV);
		this.featureVector = featureVector;
	}

	public SLLabel getLabel() {
		return label;
	}

	public Features getFeatures() {
		SLFeatureVector fvs[] = new SLFeatureVector[tagAlphabet.size()];

		for (int i = 0; i < tagAlphabet.size(); i++) {
			SLFeatureVector prodFV = new SLFeatureVector(-1, -1.0, null);

			SLFeatureVector nfv = createFeatureVector(featureVector, (String) tagAlphabet.lookupObject(i),
					new SLFeatureVector(-1, -1.0, null));

			nfv.sort();

			for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
				if (curr.index >= 0)
					prodFV = new SLFeatureVector(curr.index, 1.0, prodFV);

			fvs[i] = prodFV;
		}
		return new ClassificationFeatures(fvs);
	}

	private SLFeatureVector createFeatureVector(FeatureVector featureVector, String next, SLFeatureVector fv) {

		int[] indices = featureVector.getIndices();
		String s2 = next;
		for (int j = 0; j < indices.length; j++) {
			String pred = "feat" + indices[j];
			fv = fv.add(s2 + "_" + pred, 1.0, dataAlphabet);
		}
		return fv;
	}

	protected static void setTagAlphabet(LabelAlphabet tagAlphabet1) {
		tagAlphabet = tagAlphabet1;
	}

	protected static Alphabet getTagAlphabet() {
		return tagAlphabet;
	}

	protected static void setDataAlphabet(Alphabet dataAlphabet1) {
		dataAlphabet = dataAlphabet1;
	}

	protected static Alphabet getDataAlphabet() {
		return dataAlphabet;
	}
}
