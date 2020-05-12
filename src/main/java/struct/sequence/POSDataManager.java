/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.logging.Logger;

import struct.types.SLFeatureVector;

/**
 * POSDataManager.
 *
 * @version 07/15/2006
 */
public class POSDataManager extends SequenceDataManager {
	private static Logger logger = Logger.getLogger(POSDataManager.class.getName());

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.DataManager#readData(java.lang.String, boolean)
	 */
	@Override
	public SequenceInstance[] readData(String file, boolean createFeatureFile) throws IOException {

		BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(file), "8859_2"));
		String line = in.readLine();
		// String pos_line = in.readLine();
		String tags_line = in.readLine();

		LinkedList lt = new LinkedList();

		ObjectOutputStream out = createFeatureFile ? new ObjectOutputStream(new FileOutputStream(file + ".feats"))
				: null;

		// BufferedWriter crf = new BufferedWriter(new FileWriter(file+".crf"));

		int num1 = 0;
		logger.info("Creating feature vectors and/or forests ...");
		while (line != null) {
			if (num1 > 0 && num1 % 500 == 0)
				logger.info("Creating Feature Vector Instance: " + num1 + ", Num Feats: " + dataAlphabet.size());

			String[] toks = line.split("\t");
			// String[] pos = pos_line.split("\t");
			String[] tags = tags_line.split("\t");

			String[] toks_new = new String[toks.length + 1];
			// String[] pos_new = new String[pos.length+1];
			String[] tags_new = new String[tags.length + 1];
			toks_new[0] = "<init>";
			// pos_new[0] = "<init-POS>";
			tags_new[0] = "O";
			for (int i = 0; i < toks.length; i++) {
				toks_new[i + 1] = normalize(toks[i]);
				// pos_new[i+1] = pos[i];
				tags_new[i + 1] = tags[i];
			}
			toks = toks_new;
			// pos = pos_new;
			tags = tags_new;

			LinkedList[] predicates = getPredicates(toks);
			SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null));
			fv.sort();

			SequenceInstance si = new SequenceInstance(toks, tags, fv, predicates);

			if (createFeatureFile)
				createForest(si, predicates, out);

			// createCRF(predicates,tags,crf);

			lt.add(si);

			line = in.readLine();
			// pos_line = in.readLine();
			tags_line = in.readLine();
			num1++;
		}

		SequenceInstance[] si = new SequenceInstance[lt.size()];
		for (int i = 0; i < si.length; i++) {
			si[i] = (SequenceInstance) lt.get(i);
		}

		if (createFeatureFile)
			out.close();

		// crf.close();
		return si;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * struct.sequence.SequenceDataManager#createAlphabets(java.lang.String,
	 * boolean)
	 */
	@Override
	public void createAlphabets(String file, boolean createUnsupported) throws IOException {
		logger.info("Creating Entity Alphabet ... ");
		createTagAlphabet(file);
		logger.info("done.");
		logger.info("Num Labels: " + tagAlphabet.size());
		for (int i = 0; i < tagAlphabet.size(); i++)
			logger.info((String) tagAlphabet.lookupObject(i));

		logger.info("Creating Data Alphabet ... ");

		BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(file), "8859_2"));
		String line = in.readLine();
		// String pos_line = in.readLine();
		String tags_line = in.readLine();

		int cnt = 0;
		while (line != null) {

			String[] toks = line.split("\t");
			// String[] pos = pos_line.split("\t");
			String[] tags = tags_line.split("\t");

			String[] toks_new = new String[toks.length + 1];
			// String[] pos_new = new String[pos.length+1];
			String[] tags_new = new String[tags.length + 1];
			toks_new[0] = "<init>";
			// pos_new[0] = "<init-POS>";
			tags_new[0] = "O";
			for (int i = 0; i < toks.length; i++) {
				toks_new[i + 1] = normalize(toks[i]);
				// pos_new[i+1] = pos[i];
				tags_new[i + 1] = tags[i];
			}
			toks = toks_new;
			// pos = pos_new;
			tags = tags_new;

			LinkedList[] predicates = getPredicates(toks);
			SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null));

			if (createUnsupported)
				createU(predicates);

			line = in.readLine();
			// pos_line = in.readLine();
			tags_line = in.readLine();
			// System.out.println(cnt + ": " + dataAlphabet.size());
			cnt++;
		}

		logger.info("done.");
	}

	private void createTagAlphabet(String file) throws IOException {

		BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(file), "8859_2"));
		String line = in.readLine();
		// String pos_line = in.readLine();
		String tags_line = in.readLine();

		while (line != null) {

			String[] tags = tags_line.split("\t");

			tagAlphabet.lookupIndex("O");
			for (int i = 0; i < tags.length; i++) {
				tagAlphabet.lookupIndex(tags[i]);
			}

			line = in.readLine();
			// pos_line = in.readLine();
			tags_line = in.readLine();
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * struct.sequence.SequenceDataManager#getPredicates(java.lang.String[],
	 * java.lang.String[])
	 */
	@Override
	public LinkedList[] getPredicates(String[] toks, String[] pos) {
		return getPredicates(toks);
	}

	public LinkedList[] getPredicates(String[] toks) {
		String ALPHA = "[A-Za-z]";
		String ALPHANUM = "[A-Za-z0-9]";
		String PUNT = "[,\\.;:?!]";

		LinkedList[] predicates = new LinkedList[toks.length];

		for (int i = 0; i < toks.length; i++) {
			predicates[i] = new LinkedList();

			String word = toks[i];

			// ortho predicates
			if (word.matches("[ivxdlcm]+|[IVXDLCM]+"))
				predicates[i].add("ORT=ROMAN");
			if (word.matches("\\.\\.+"))
				predicates[i].add("ORT=MULTIDOT");
			if (word.matches("[^\\.]+.*\\."))
				predicates[i].add("ORT=ENDDOT");
			if (word.matches(ALPHANUM + "+-" + ALPHANUM + "*"))
				predicates[i].add("ORT=DASH");
			if (word.matches(ALPHA))
				predicates[i].add("ORT=SING");
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
			if (i > 1) {
				predicates[i].add("WORD-2=" + toksL[i - 2]);
				predicates[i].add("WORD=" + toksL[i] + "_WORD-2=" + toksL[i - 2]);
			}
			if (i < toks.length - 2) {
				predicates[i].add("WORD+2=" + toksL[i + 2]);
				predicates[i].add("WORD=" + toksL[i] + "_WORD+2=" + toksL[i + 2]);
			}
			for (int j = 2; j < 5; j++) {
				if (toksL[i].length() > j) {
					String pref = toksL[i].substring(0, j);
					String suff = toksL[i].substring(toksL[i].length() - j, toksL[i].length());
					predicates[i].add("PREF" + j + "=" + pref);
					predicates[i].add("SUFF" + j + "=" + suff);
					if (i > 0) {
						String pref_n1 = toksL[i - 1].length() > j ? toksL[i - 1].substring(0, j) : toks[i - 1];
						String suff_n1 = toksL[i - 1].length() > j ? toksL[i - 1].substring(toksL[i - 1].length() - j)
								: toks[i - 1];
						predicates[i].add("PREF" + j + "=" + pref + "_PREF-1" + j + "=" + pref_n1);
						predicates[i].add("SUFF" + j + "=" + suff + "_PREF-1" + j + "=" + pref_n1);
						predicates[i].add("PREF" + j + "=" + pref + "_SUFF-1" + j + "=" + suff_n1);
						predicates[i].add("SUFF" + j + "=" + suff + "_SUFF-1" + j + "=" + suff_n1);
					}
				}
			}
		}
		return predicates;
	}
}
