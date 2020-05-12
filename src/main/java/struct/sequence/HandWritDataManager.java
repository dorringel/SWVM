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

import cc.mallet.types.Alphabet;
import struct.types.DataManager;
import struct.types.SLFeatureVector;

/**
 * Hand written DataManager.
 *
 * @version 08/15/2006
 */
public class HandWritDataManager implements DataManager {
	private static Logger logger = Logger.getLogger(HandWritDataManager.class.getName());
	private final Alphabet dataAlphabet;
	private final Alphabet tagAlphabet;
	private final int degree;

	private double tran_val, node_val;

	public HandWritDataManager(int degree) {
		dataAlphabet = new Alphabet();
		tagAlphabet = new Alphabet();
		this.degree = degree;
		if (degree == 2) {
			tran_val = 29.0;
			node_val = Math.sqrt(2.0);
		} else {
			tran_val = 5.29;
			node_val = 1.0;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.DataManager#getDataAlphabet()
	 */
	public Alphabet getDataAlphabet() {
		return dataAlphabet;
	}

	public Alphabet getTagAlphabet() {
		return tagAlphabet;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.DataManager#readData(java.lang.String)
	 */
	public SequenceInstance[] readData(String file) throws IOException {
		return readData(file, true);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.DataManager#readData(java.lang.String, boolean)
	 */
	public SequenceInstance[] readData(String file, boolean createFeatureFile) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(file), "8859_2"));
		String line = in.readLine();
		String tags_line = in.readLine();

		LinkedList lt = new LinkedList();

		ObjectOutputStream out = createFeatureFile ? new ObjectOutputStream(new FileOutputStream(file + ".feats"))
				: null;

		int num1 = 0;
		int numToks = 0;
		logger.info("Creating feature vectors and/or forests ...");
		while (line != null) {
			if (num1 > 0 && num1 % 500 == 0)
				logger.info("Creating Feature Vector Instance: " + num1 + ", Num Feats: " + dataAlphabet.size());

			String[] toks = line.split("\t");
			String[] tags = tags_line.split("\t");
			numToks += toks.length;

			String[] toks_new = new String[toks.length + 1];
			String[] tags_new = new String[tags.length + 1];
			toks_new[0] = "0";
			tags_new[0] = "O";
			for (int i = 0; i < toks.length; i++) {
				toks_new[i + 1] = normalize(toks[i]);
				tags_new[i + 1] = tags[i];
			}
			toks = toks_new;
			tags = tags_new;

			LinkedList[] predicates = getPredicates(toks);
			SLFeatureVector fv = createFeatureVector(predicates, tags, new SLFeatureVector(-1, -1.0, null));
			fv.sort();

			SequenceInstance si = new HandWritInstance(toks, tags, fv, predicates);

			if (createFeatureFile)
				createForest(si, predicates, out);

			lt.add(si);

			line = in.readLine();
			tags_line = in.readLine();
			num1++;
		}

		SequenceInstance[] si = new SequenceInstance[lt.size()];
		for (int i = 0; i < si.length; i++) {
			si[i] = (SequenceInstance) lt.get(i);
		}

		if (createFeatureFile)
			out.close();

		logger.info("Num toks: " + numToks);
		return si;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.DataManager#createAlphabets(java.lang.String)
	 */
	public void createAlphabets(String file) throws IOException {
		createAlphabets();
	}

	/**
	 * Creates the tag and the data alphabets.
	 */
	public void createAlphabets() {
		logger.info("Creating Entity Alphabet ... ");
		createTagAlphabet();
		logger.info("done.");
		logger.info("Num Labels: " + tagAlphabet.size());

		logger.info("Creating Data Alphabet ... ");
		for (int i = 0; i < 26; i++) {
			String prev = Character.toString((char) ('a' + i));
			for (int j = 0; j < 26; j++) {
				String next = Character.toString((char) ('a' + j));
				dataAlphabet.lookupIndex(prev + "_" + next);
			}
			for (int j = 0; j < 128; j++)
				dataAlphabet.lookupIndex(prev + "_" + j);
		}
		for (int i = 0; i < 26 && degree == 1; i++) {
			String prev = Character.toString((char) ('a' + i));
			for (int j = 0; j < 128; j++)
				dataAlphabet.lookupIndex(prev + "_" + j);
		}
		for (int i = 0; i < 26 && degree == 2; i++) {
			String prev = Character.toString((char) ('a' + i));
			for (int j = 0; j < 128; j++) {
				dataAlphabet.lookupIndex(prev + "_" + j);
				for (int k = j + 1; k < 128; k++)
					dataAlphabet.lookupIndex(prev + "_" + j + "_" + k);
			}
		}
	}

	private void createTagAlphabet() {

		for (int i = 0; i < 26; i++) {
			tagAlphabet.lookupIndex(Character.toString((char) ('a' + i)));
		}
		tagAlphabet.lookupIndex("O");
	}

	private SLFeatureVector createFeatureVector(LinkedList predicates, String prev, String next, SLFeatureVector fv) {

		String s1 = prev + "_" + next;
		fv = fv.add(s1, tran_val, dataAlphabet);

		return fv;
	}

	private SLFeatureVector createFeatureVector(LinkedList predicates, String next, SLFeatureVector fv) {

		String s2 = next;
		if (predicates.size() > 0) {
			String pred = (String) predicates.get(0);
			fv = fv.add(s2 + "_" + pred, 1.0, dataAlphabet);
		}
		for (int j = 1; j < predicates.size(); j++) {
			String pred = (String) predicates.get(j);
			fv = fv.add(s2 + "_" + pred, node_val, dataAlphabet);
		}

		return fv;
	}

	private SLFeatureVector createFeatureVector(LinkedList[] predicates, String[] tags, SLFeatureVector fv) {

		for (int i = 1; i < predicates.length; i++) {
			fv = createFeatureVector(predicates[i], tags[i - 1], tags[i], fv);
			fv = createFeatureVector(predicates[i], tags[i], fv);
		}

		return fv;
	}

	public LinkedList[] getPredicates(String[] toks) {
		LinkedList[] predicates = new LinkedList[toks.length];
		for (int i = 0; i < toks.length; i++) {
			predicates[i] = new LinkedList();
			String[] ts = toks[i].split(" ");
			for (int j = 0; j < ts.length; j++)
				predicates[i].add(ts[j]);
			if (degree == 2) {
				for (int j = 0; j < ts.length; j++) {
					for (int k = j + 1; k < ts.length; k++)
						predicates[i].add(ts[j] + "_" + ts[k]);
				}
			}
		}
		return predicates;
	}

	/**
	 * Stops alphabets' growths.
	 */
	public void closeAlphabets() {
		dataAlphabet.stopGrowth();
		tagAlphabet.stopGrowth();
		SequenceInstance.setTagAlphabet(tagAlphabet);
		SequenceInstance.setDataAlphabet(dataAlphabet);
	}

	private void createForest(SequenceInstance inst, LinkedList[] predicates, ObjectOutputStream out) {
		String[] toks = inst.getInput().sentence;

		try {

			for (int k = 0; k < predicates.length; k++) {
				for (int i = 0; i < tagAlphabet.size(); i++) {
					for (int j = 0; j < tagAlphabet.size(); j++) {
						SLFeatureVector nfv = createFeatureVector(predicates[k], (String) tagAlphabet.lookupObject(i),
								(String) tagAlphabet.lookupObject(j), new SLFeatureVector(-1, -1.0, null));

						nfv.sort();

						for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
							if (curr.index >= 0) {
								out.writeInt(curr.index);
								out.writeDouble(curr.value);
							}
						out.writeInt(-1);
					}
				}
			}

			out.writeInt(-2);

			for (int k = 0; k < predicates.length; k++) {
				for (int j = 0; j < tagAlphabet.size(); j++) {
					SLFeatureVector nfv = createFeatureVector(predicates[k], (String) tagAlphabet.lookupObject(j),
							new SLFeatureVector(-1, -1.0, null));

					nfv.sort();

					for (SLFeatureVector curr = nfv; curr.next != null; curr = curr.next)
						if (curr.index >= 0) {
							out.writeInt(curr.index);
							out.writeDouble(curr.value);
						}
					out.writeInt(-1);
				}
			}

			out.writeInt(-3);

		} catch (IOException e) {
		}
	}

	public String normalize(String s) {
		if (s.matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"))
			return "<num>";

		return s;
	}
}
