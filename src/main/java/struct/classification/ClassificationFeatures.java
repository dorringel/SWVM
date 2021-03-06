/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.classification;

import struct.types.Features;
import struct.types.SLFeatureVector;

/**
 * Features of a classification instance.
 *
 * @version 08/15/2006
 */
public class ClassificationFeatures implements Features {
	
	//One SLFeatureVector per label
    private SLFeatureVector[] fvs;
    
    public ClassificationFeatures(SLFeatureVector[] fvs) {
    	this.fvs = fvs;
    }
      
    public SLFeatureVector getFeatureVector(int n) {
    	return fvs[n];
    }
    
    public int getFeaturesSize() {
    	return fvs.length;
    }
}
