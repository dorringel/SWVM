/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import struct.types.Input;

/**
 * A sequence input.
 *
 * @version 07/15/2006
 */
public class SequenceInput implements Input {

	public String[] sentence;
	public String[] pos;

	public SequenceInput(String[] sentence, String[] pos) {
		this.sentence = sentence;
		this.pos = pos;
	}

	public SequenceInput(String[] sentence) {
		this.sentence = sentence;
		this.pos = null;
	}
}
