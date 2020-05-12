/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.util.LinkedList;

/**
 * ChunkerDataManager.
 *
 * @version 08/15/2006
 */
public class ChunkerDataManager extends SequenceDataManager {

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * struct.sequence.SequenceDataManager#getPredicates(java.lang.String[],
	 * java.lang.String[])
	 */
	@Override
	public LinkedList[] getPredicates(String[] toks, String[] pos) {
		String CAPS = "[A-Z]";
		String LOW = "[a-z]";
		String CAPSNUM = "[A-Z0-9]";
		String ALPHA = "[A-Za-z]";
		String ALPHANUM = "[A-Za-z0-9]";
		String PUNT = "[,\\.;:?!]";

		LinkedList[] predicates = new LinkedList[toks.length];

		for (int i = 0; i < toks.length; i++) {
			predicates[i] = new LinkedList();

			// tag triples
			String tm2 = i >= 2 ? pos[i - 2] : "<SP>";
			String tm1 = i >= 1 ? pos[i - 1] : "<SP>";
			String tm0 = pos[i];
			String tp2 = i < pos.length - 2 ? pos[i + 2] : "<EP>";
			String tp1 = i < pos.length - 1 ? pos[i + 1] : "<EP>";

			String wm2 = i >= 2 ? toks[i - 2] : "<SW>";
			String wm1 = i >= 1 ? toks[i - 1] : "<SW>";
			String wm0 = toks[i];
			String wp2 = i < toks.length - 2 ? toks[i + 2] : "<EW>";
			String wp1 = i < toks.length - 1 ? toks[i + 1] : "<EW>";

			String wsin0 = "W0=" + wm2;
			String wsin1 = "W1=" + wm1;
			String wsin2 = "W2=" + wm0;
			String wsin3 = "W3=" + wp1;
			String wsin4 = "W4=" + wp2;

			String tsin0 = "T0=" + tm2;
			String tsin1 = "T1=" + tm1;
			String tsin2 = "T2=" + tm0;
			String tsin3 = "T3=" + tp1;
			String tsin4 = "T4=" + tp2;

			String wbin0 = "WW0=" + wm2 + "_" + wm1;
			String wbin1 = "WW1=" + wm1 + "_" + wm0;
			String wbin2 = "WW2=" + wm0 + "_" + wp1;
			String wbin3 = "WW3=" + wp1 + "_" + wp2;

			String tbin0 = "TT0=" + tm2 + "_" + tm1;
			String tbin1 = "TT1=" + tm1 + "_" + tm0;
			String tbin2 = "TT2=" + tm0 + "_" + tp1;
			String tbin3 = "TT3=" + tp1 + "_" + tp2;

			String ttrip0 = "TTT0=" + tm2 + "_" + tm1 + "_" + tm0;
			String ttrip1 = "TTT1=" + tm1 + "_" + tm0 + "_" + tp1;
			String ttrip2 = "TTT2=" + tm0 + "_" + tp1 + "_" + tp2;

			predicates[i].add(wsin0);
			predicates[i].add(wsin1);
			predicates[i].add(wsin2);
			predicates[i].add(wsin3);
			predicates[i].add(wsin4);

			predicates[i].add(tsin0);
			predicates[i].add(tsin1);
			predicates[i].add(tsin2);
			predicates[i].add(tsin3);
			predicates[i].add(tsin4);

			predicates[i].add(wbin0);
			predicates[i].add(wbin1);
			predicates[i].add(wbin2);
			predicates[i].add(wbin3);

			predicates[i].add(tbin0);
			predicates[i].add(tbin1);
			predicates[i].add(tbin2);
			predicates[i].add(tbin3);

			predicates[i].add(ttrip0);
			predicates[i].add(ttrip1);
			predicates[i].add(ttrip2);
		}
		return predicates;
	}
}
