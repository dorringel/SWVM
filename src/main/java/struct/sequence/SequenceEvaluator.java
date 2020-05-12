/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.sequence;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.text.DecimalFormat;
import java.util.Scanner;
import java.util.logging.Logger;

import cc.mallet.types.Alphabet;
import gnu.trove.TIntObjectHashMap;
import gnu.trove.TObjectIntHashMap;
import struct.alg.Predictor;
import struct.types.Features;
import struct.types.SLEvaluator;
import struct.types.SLInstance;

/**
 * A sequence evaluator.
 *
 * @version 07/15/2006
 */
public class SequenceEvaluator implements SLEvaluator {
	private static Logger logger = Logger.getLogger(SequenceEvaluator.class.getName());

	private final Alphabet tagAlphabet;

	public String outFile = null;

	public SequenceEvaluator(Alphabet tagAlphabet) {
		this.tagAlphabet = tagAlphabet;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.Evaluator#evaluate(struct.types.Instance[],
	 * struct.alg.Predictor)
	 */
	public void evaluate(SLInstance[] data, Predictor predictor) {
		TObjectIntHashMap sufMap = new TObjectIntHashMap();
		TIntObjectHashMap revSufMap = new TIntObjectHashMap();

		int nn = 0;
		for (int i = 0; i < tagAlphabet.size(); i++) {
			String tag = (String) tagAlphabet.lookupObject(i);
			String[] toks = tag.split("-");
			if (toks.length > 1) {
				String suf = toks[1];
				if (!sufMap.contains(suf)) {
					sufMap.put(suf.trim(), nn);
					revSufMap.put(nn, suf.trim());
					nn++;
				}
			}
		}

		int guessed[] = new int[sufMap.size()];
		int real[] = new int[sufMap.size()];
		int correct[] = new int[sufMap.size()];

		int acc_corr = 0;
		int acc_tot = 0;

		logger.info("Evaluating ... ");

		// ObjectInputStream in = new ObjectInputStream(new
		// FileInputStream(featureFile));

		// BufferedWriter out = outFile == null ? null : new BufferedWriter(new
		// FileWriter(outFile));

		int myTotal = 0;
		int myCorrect = 0;

		Scanner sc = new Scanner(System.in);

		for (int k = 0; k < data.length; k++) {

			SLInstance inst = data[k];
			// Features feats = data[k].getFeatures(in);
			Features feats = data[k].getFeatures();

			SequencePrediction prediction = (SequencePrediction) predictor.decode(inst, feats);

			String[] act_spans = ((SequenceLabel) inst.getLabel()).tags;
			String[] pred_spans = prediction.getBestLabel().tags;

			if (k % 1 == 0)
				for (int i = 0; i < pred_spans.length; i++)
					logger.info(pred_spans[i] + " ");

			myTotal += act_spans.length;
			for (int i = 0; i < act_spans.length; i++) {
				if (act_spans[i].equals(pred_spans[i]))
					myCorrect++;
			}

			if (k % 1 == 0) {
				logger.info("\n");
				logger.info("MyAccuracy=" + ((double) myCorrect) / myTotal);
				sc.nextLine();
			}

			for (int i = 0; i < pred_spans.length; i++) {
				String pred = pred_spans[i];
				String act = act_spans[i];
				if (pred.equals(act))
					acc_corr++;
				acc_tot++;

				if (pred.startsWith("B")) {
					String pred_suf = pred.split("-")[1];
					guessed[sufMap.get(pred_suf)]++;

				}
				if (act.startsWith("B")) {
					String act_suf = act.split("-")[1];
					real[sufMap.get(act_suf)]++;
				}
				if (pred.startsWith("B") && act.startsWith("B") && pred.equals(act)) {

					String pred_suf = pred.split("-")[1];

					String cont = "I-" + pred_suf;

					for (int j = i + 1; j < pred_spans.length; j++) {
						pred = pred_spans[j];
						act = act_spans[j];
						if (pred.equals(act) && pred.equals(cont) && j == pred_spans.length - 1) {
							correct[sufMap.get(pred_suf)]++;
							break;
						}
						if (pred.equals(cont) && !act.equals(cont)) {
							break;
						}
						if (!pred.equals(cont) && act.equals(cont))
							break;
						if (!pred.equals(cont) && !act.equals(cont)) {
							correct[sufMap.get(pred_suf)]++;
							break;
						}
					}

					if (i == pred_spans.length - 1)
						correct[sufMap.get(pred_suf)]++;

				}

			}
		}

		logger.info("MyAccuracy=" + ((double) myCorrect) / myTotal);

		DecimalFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);

		int tot_corr = 0;
		int tot_guessed = 0;
		int tot_real = 0;
		for (int i = 0; i < correct.length; i++) {
			logger.info(revSufMap.get(i) + ":");
			logger.info("\tGuessed: " + guessed[i] + ", Actual: " + real[i] + ", Correct: " + correct[i]);
			double prec = (double) correct[i] / guessed[i];
			double rec = (double) correct[i] / real[i];
			double fmeas = (2 * prec * rec) / (prec + rec);
			logger.info("\tPrecision: " + nf.format(prec) + ", Recall: " + nf.format(rec) + ", F-meas: "
					+ nf.format(fmeas));
			tot_corr += correct[i];
			tot_real += real[i];
			tot_guessed += guessed[i];
		}
		logger.info("Total:");
		logger.info("\tGuessed: " + tot_guessed + ", Actual: " + tot_real + ", Correct: " + tot_corr);
		double prec = (double) tot_corr / tot_guessed;
		double rec = (double) tot_corr / tot_real;
		double fmeas = (2 * prec * rec) / (prec + rec);
		logger.info(
				"\tPrecision: " + nf.format(prec) + ", Recall: " + nf.format(rec) + ", F-meas: " + nf.format(fmeas));

		logger.info("done.");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.Evaluator#evaluate(struct.types.Instance[],
	 * struct.alg.Predictor, java.lang.String)
	 */
	public void evaluate(SLInstance[] data, Predictor predictor, String featureFile) throws IOException {
		TObjectIntHashMap sufMap = new TObjectIntHashMap();
		TIntObjectHashMap revSufMap = new TIntObjectHashMap();

		int nn = 0;
		for (int i = 0; i < tagAlphabet.size(); i++) {
			String tag = (String) tagAlphabet.lookupObject(i);
			String[] toks = tag.split("-");
			if (toks.length > 1) {
				String suf = toks[1];
				if (!sufMap.contains(suf)) {
					sufMap.put(suf.trim(), nn);
					revSufMap.put(nn, suf.trim());
					nn++;
				}
			}
		}

		int guessed[] = new int[sufMap.size()];
		int real[] = new int[sufMap.size()];
		int correct[] = new int[sufMap.size()];

		int acc_corr = 0;
		int acc_tot = 0;

		logger.info("Evaluating ... ");

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(featureFile));

		BufferedWriter out = outFile == null ? null : new BufferedWriter(new FileWriter(outFile));

		for (int k = 0; k < data.length; k++) {

			SLInstance inst = data[k];
			// Features feats = data[k].getFeatures(in);
			Features feats = data[k].getFeatures();

			SequencePrediction prediction = (SequencePrediction) predictor.decode(inst, feats);

			String[] act_spans = ((SequenceLabel) inst.getLabel()).tags;
			String[] pred_spans = prediction.getBestLabel().tags;

			if (out != null) {
				String res = "";
				for (int i = 1; i < pred_spans.length; i++)
					res += pred_spans[i] + "\t";
				res = res.trim();
				out.write(res + "\n");
			}

			for (int i = 0; i < pred_spans.length; i++) {
				String pred = pred_spans[i];
				String act = act_spans[i];
				if (pred.equals(act))
					acc_corr++;
				acc_tot++;

				if (pred.startsWith("B")) {
					String pred_suf = pred.split("-")[1];
					guessed[sufMap.get(pred_suf)]++;

				}
				if (act.startsWith("B")) {
					String act_suf = act.split("-")[1];
					real[sufMap.get(act_suf)]++;
				}
				if (pred.startsWith("B") && act.startsWith("B") && pred.equals(act)) {

					String pred_suf = pred.split("-")[1];

					String cont = "I-" + pred_suf;

					for (int j = i + 1; j < pred_spans.length; j++) {
						pred = pred_spans[j];
						act = act_spans[j];
						if (pred.equals(act) && pred.equals(cont) && j == pred_spans.length - 1) {
							correct[sufMap.get(pred_suf)]++;
							break;
						}
						if (pred.equals(cont) && !act.equals(cont)) {
							break;
						}
						if (!pred.equals(cont) && act.equals(cont))
							break;
						if (!pred.equals(cont) && !act.equals(cont)) {
							correct[sufMap.get(pred_suf)]++;
							break;
						}
					}

					if (i == pred_spans.length - 1)
						correct[sufMap.get(pred_suf)]++;

				}

			}
		}

		if (out != null)
			out.close();

		in.close();
		DecimalFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);

		int tot_corr = 0;
		int tot_guessed = 0;
		int tot_real = 0;
		for (int i = 0; i < correct.length; i++) {
			logger.info(revSufMap.get(i) + ":");
			logger.info("\tGuessed: " + guessed[i] + ", Actual: " + real[i] + ", Correct: " + correct[i]);
			double prec = (double) correct[i] / guessed[i];
			double rec = (double) correct[i] / real[i];
			double fmeas = (2 * prec * rec) / (prec + rec);
			logger.info("\tPrecision: " + nf.format(prec) + ", Recall: " + nf.format(rec) + ", F-meas: "
					+ nf.format(fmeas));
			tot_corr += correct[i];
			tot_real += real[i];
			tot_guessed += guessed[i];
		}
		logger.info("Total:");
		logger.info("\tGuessed: " + tot_guessed + ", Actual: " + tot_real + ", Correct: " + tot_corr);
		double prec = (double) tot_corr / tot_guessed;
		double rec = (double) tot_corr / tot_real;
		double fmeas = (2 * prec * rec) / (prec + rec);
		logger.info(
				"\tPrecision: " + nf.format(prec) + ", Recall: " + nf.format(rec) + ", F-meas: " + nf.format(fmeas));

		logger.info("done.");
	}
}
