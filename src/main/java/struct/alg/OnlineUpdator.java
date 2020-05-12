/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.alg;

import struct.types.Features;
import struct.types.SLInstance;

/** An updator updates the predictor.
 * 
 * @version 08/15/2006
 */
public interface OnlineUpdator {
	
	/** Updates the predictor based on the instance and its features.
	 */
    public void update(SLInstance inst, Features feats, Predictor pred, double avg_upd);
}
