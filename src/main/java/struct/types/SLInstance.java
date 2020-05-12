/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.types;

/** A generic instance.
 * 
 * @version 08/15/2006
 */
public interface SLInstance {

    //public Input getInput();
    
	/** Returns the target label.
	 */
	public SLLabel getLabel();

	/** Returns the features.
	 */
    public Features getFeatures();

    //public Features getFeatures(ObjectInputStream in) throws IOException;

}
