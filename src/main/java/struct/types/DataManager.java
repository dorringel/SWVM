/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.types;

import java.io.IOException;

import cc.mallet.types.Alphabet;

/**
 * A generic data manager.
 *
 * @version 08/15/2006
 */
public interface DataManager {

	/**
	 * Creates alphabets by reading from the file according to this data
	 * manager.
	 *
	 * @throws IOException
	 */
	public void createAlphabets(String file) throws IOException;

	/**
	 * Stops alphabets' growths.
	 */
	public void closeAlphabets();

	/**
	 * Creates instances by reading from the file according to this data
	 * manager.
	 * 
	 * @throws IOException
	 */
	public SLInstance[] readData(String file) throws IOException;

	/**
	 * Returns data alphabet.
	 * 
	 */
	public Alphabet getDataAlphabet();
}
