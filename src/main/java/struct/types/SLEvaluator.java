/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.types;

import java.io.*;

import struct.alg.*;

/** A generic evaluator.
 * 
 * @version 08/15/2006
 */
public interface SLEvaluator {

	/** Evaluates the predictor with the data
	 * according to this evaluator.
	 */
    public void evaluate(SLInstance[] data, Predictor pred);

    /** Evaluates the predictor with the data
	 * according to this evaluator.
     * 
     * @throws IOException
     */
    public void evaluate(SLInstance[] data, Predictor pred, String featureFile) throws IOException;
}
