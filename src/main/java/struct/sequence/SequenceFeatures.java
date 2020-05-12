/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import struct.types.Features;
import struct.types.SLFeatureVector;

/**
 * Features for sequences.
 *
 * @version 07/15/2006
 */
public class SequenceFeatures implements Features {

	private final SLFeatureVector[][][] fvs_ij;
	private final SLFeatureVector[][] fvs_j;

	/**
	 *
	 * @param fvs_ij
	 *            Features for position-label-label
	 * @param fvs_j
	 *            Features for position-label
	 */
	public SequenceFeatures(SLFeatureVector[][][] fvs_ij, SLFeatureVector[][] fvs_j) {
		this.fvs_ij = fvs_ij;
		this.fvs_j = fvs_j;
	}

	/**
	 * Gets the SLFeatureVector by position-label-label.
	 * 
	 * @param n
	 *            Position index
	 * @param y1
	 *            Index of first label
	 * @param y2
	 *            Index of second label
	 */
	public SLFeatureVector getFeatureVector(int n, int y1, int y2) {
		return fvs_ij[n][y1][y2];
	}

	/**
	 * Gets the SLFeatureVector by position-label.
	 *
	 * @param n
	 *            Position index
	 * @param y1
	 *            Index of label
	 */
	public SLFeatureVector getFeatureVector(int n, int y1) {
		return fvs_j[n][y1];
	}
}
