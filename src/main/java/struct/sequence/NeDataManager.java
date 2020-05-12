/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.util.LinkedList;

/**
 * NeDataManager.
 *
 * @version 07/15/2006
 */
public class NeDataManager extends SequenceDataManager {

	private final boolean usePOS = true;

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

			String word = toks[i];

			// ortho predicates
			if (word.matches(CAPS + ".*"))
				predicates[i].add("ORT=INITCAP");
			if (word.matches(CAPS + LOW + "*"))
				predicates[i].add("ORT=CAPS");
			if (word.matches(CAPS + "+"))
				predicates[i].add("ORT=ALLCAPS");
			if (word.matches("[ivxdlcm]+|[IVXDLCM]+"))
				predicates[i].add("ORT=ROMAN");
			if (word.matches("\\.\\.+"))
				predicates[i].add("ORT=MULTIDOT");
			if (word.matches("[^\\.]+.*\\."))
				predicates[i].add("ORT=ENDDOT");
			if (word.matches(ALPHANUM + "+-" + ALPHANUM + "*"))
				predicates[i].add("ORT=DASH");
			if (word.matches("[A-Z][A-Z\\.]*\\.[A-Z\\.]*"))
				predicates[i].add("ORT=ACRO");
			if (word.matches(CAPS + "\\."))
				predicates[i].add("ORT=INIT");
			if (word.matches(ALPHA))
				predicates[i].add("ORT=SING");
			if (word.matches(CAPS))
				predicates[i].add("ORT=CAP1");
			if (word.matches(PUNT))
				predicates[i].add("ORT=PUNC");

			String[] toksL = new String[toks.length];
			for (int j = 0; j < toks.length; j++)
				toksL[j] = toks[j].toLowerCase();

			// word and suffix features
			predicates[i].add("WORD=" + toksL[i]);
			if (i > 0) {
				predicates[i].add("WORD-1=" + toksL[i - 1]);
				predicates[i].add("WORD=" + toksL[i] + "_WORD-1=" + toksL[i - 1]);
			}
			if (i < toks.length - 1) {
				predicates[i].add("WORD+1=" + toksL[i + 1]);
				predicates[i].add("WORD=" + toksL[i] + "_WORD+1=" + toksL[i + 1]);
			}
			for (int j = 2; j < 5; j++) {
				if (toksL[i].length() > j) {
					predicates[i].add("PREF" + j + "=" + toksL[i].substring(0, j));
					predicates[i].add("SUFF" + j + "=" + toksL[i].substring(toksL[i].length() - j, toksL[i].length()));
				}
			}

			if (usePOS) {
				// pos features
				predicates[i].add("POS=" + pos[i]);
				predicates[i].add("POS=" + pos[i] + "_WORD=" + toksL[i]);
				if (i > 0) {
					predicates[i].add("POS-1=" + pos[i - 1]);
					predicates[i].add("POS=" + pos[i] + "_POS-1=" + pos[i - 1]);
					predicates[i].add("POS-1=" + pos[i - 1] + "_WORD=" + toksL[i]);
					predicates[i].add("POS=" + pos[i] + "_POS-1=" + pos[i - 1] + "_WORD=" + toksL[i]);
				}
				if (i > 1) {
					predicates[i].add("POS-2=" + pos[i - 2]);
					predicates[i].add("POS=" + pos[i] + "_POS-2=" + pos[i - 2]);
					predicates[i].add("POS-2=" + pos[i - 2] + "_WORD=" + toksL[i]);
					predicates[i].add("POS=" + pos[i] + "_POS-2=" + pos[i - 2] + "_WORD=" + toksL[i]);
					predicates[i].add("POS=" + pos[i] + "_POS-1=" + pos[i - 1] + "_POS-2=" + pos[i - 2]);
				}
				if (i < toks.length - 1) {
					predicates[i].add("POS+1=" + pos[i + 1]);
					predicates[i].add("POS=" + pos[i] + "_POS+1=" + pos[i + 1]);
					predicates[i].add("POS+1=" + pos[i + 1] + "_WORD=" + toksL[i]);
					predicates[i].add("POS=" + pos[i] + "_POS+1=" + pos[i + 1] + "_WORD=" + toksL[i]);
				}
				if (i < toks.length - 2) {
					predicates[i].add("POS+2=" + pos[i + 2]);
					predicates[i].add("POS=" + pos[i] + "_POS+2=" + pos[i + 2]);
					predicates[i].add("POS+2=" + pos[i + 2] + "_WORD=" + toksL[i]);
					predicates[i].add("POS=" + pos[i] + "_POS+2=" + pos[i + 2] + "_WORD=" + toksL[i]);
					predicates[i].add("POS=" + pos[i] + "_POS+1=" + pos[i + 1] + "_POS+2=" + pos[i + 2]);
				}
				if (i > 0 && i < toks.length - 1)
					predicates[i].add("POS=" + pos[i] + "_POS+1=" + pos[i + 1] + "_POS-1=" + pos[i - 1]);
			}
		}
		return predicates;
	}
}
