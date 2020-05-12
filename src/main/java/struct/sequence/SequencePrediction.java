/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import struct.types.Prediction;

/**
 * Prediction for a sequence.
 *
 * @version 07/15/2006
 */
public class SequencePrediction implements Prediction {

	private final SequenceLabel[] labels;

	public SequencePrediction(SequenceLabel[] labels) {
		this.labels = labels;
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see struct.types.Prediction#getBestLabel()
	 */
	@Override
	public SequenceLabel getBestLabel() {
		return labels[0];
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see struct.types.Prediction#getLabelByRank(int)
	 */
	@Override
	public SequenceLabel getLabelByRank(int rank) {
		if (rank >= labels.length)
			return null;
		return labels[rank];
	}

	public int getNumLabels() {
		return labels.length;
	}
}
