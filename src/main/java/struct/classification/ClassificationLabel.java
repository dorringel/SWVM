/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.classification;

import struct.types.SLFeatureVector;
import struct.types.SLLabel;

/**
 * Internal representation of the target label of a classification instance.
 *
 * @version 08/20/2006
 */
public class ClassificationLabel implements SLLabel {
	// The string representing this label
	private final String tag;
	// The SLFeatureVector specific to this label
	private final SLFeatureVector fv;

	/**
	 * @param tag
	 *            - The string representing the target label.
	 * @param fv
	 *            - The SLFeatureVector containing all the features specific to
	 *            this label.
	 */
	public ClassificationLabel(String tag, SLFeatureVector fv) {
		this.tag = tag;
		this.fv = fv;
	}

	public SLFeatureVector getFeatureVectorRepresentation() {
		return fv;
	}

	/**
	 * Evaluates a prediction against this label
	 */
	public double loss(SLLabel pred) {
		return hammingDistance(pred);
	}

	private int hammingDistance(SLLabel pred) {
		ClassificationLabel spred = (ClassificationLabel) pred;

		String pred_tag = spred.tag;

		if (!tag.equals(pred_tag))
			return 1;
		else
			return 0;
	}

	public String getTag() {
		return tag;
	}
}
