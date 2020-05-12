/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import struct.types.SLFeatureVector;
import struct.types.SLLabel;

/** A sequence label.
 * 
 * @version 07/15/2006
 */
public class SequenceLabel implements SLLabel {
	public String[] tags;
	public SLFeatureVector fv;
	
	public SequenceLabel(String[] tags, SLFeatureVector fv) {		
		this.tags = tags;
		this.fv = fv;		
	}
	
	/*
	 *  (non-Javadoc)
	 * @see struct.types.Label#getFeatureVectorRepresentation()
	 */
	public SLFeatureVector getFeatureVectorRepresentation() { 
		return fv; 
	}
	
	/*	Evaluates a prediction against this label
	 *  (non-Javadoc)
	 * @see struct.types.Label#loss(struct.types.Label)
	 */
	public double loss(SLLabel pred) {		
		return (double)hammingDistance(pred);		
	}
	
	/** Computes hamming distance.
	 */
	public int hammingDistance(SLLabel pred) {
		SequenceLabel spred = (SequenceLabel)pred;
		
		String[] act_tags = tags;
		String[] pred_tags = spred.tags;
		
		int numErrors = 0;
		
		for(int i = 0; i < act_tags.length; i++) {
			if(!act_tags[i].equals(pred_tags[i]))
				numErrors++;
		}
		
		return numErrors;
		
	}
	
	/**1 - hammingDistance
	 */
	public int correct(SLLabel pred) { 
		return tags.length - hammingDistance(pred); 
	}	
}
