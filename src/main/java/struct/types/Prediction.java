/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.types;

/** A generic prediction.
 * 
 * @version 08/15/2006
 */
public interface Prediction {
	
	/** Returns the best label according to
	 * this prediction.
	 */
    public SLLabel getBestLabel();
    
    /** Returns the label by rank according to
	 * this prediction.
     */
    public SLLabel getLabelByRank(int rank);
}
