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
import java.util.logging.Logger;

import struct.alg.Predictor;
import struct.types.Features;
import struct.types.SLEvaluator;
import struct.types.SLInstance;

/**
 * An evaluator to evaluate sequence accuracy.
 *
 * @version 07/15/2006
 */
public class SequenceAccuracyEvaluator implements SLEvaluator {
	private static Logger logger = Logger.getLogger(SequenceAccuracyEvaluator.class.getName());

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.Evaluator#evaluate(struct.types.Instance[],
	 * struct.alg.Predictor)
	 */
	public void evaluate(SLInstance[] data, Predictor pred) {
		logger.severe("Not implemented.");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see struct.types.Evaluator#evaluate(struct.types.Instance[],
	 * struct.alg.Predictor, java.lang.String)
	 */
	public void evaluate(SLInstance[] data, Predictor predictor, String featureFile) throws IOException {
		evaluate(data, predictor, featureFile, null);
	}

	/**
	 * Evaluates the predictor with the instances.
	 * 
	 * @throws IOException
	 */
	public void evaluate(SLInstance[] data, Predictor predictor, String featureFile, String outFile)
			throws IOException {
		int acc_corr = 0;
		int acc_tot = 0;

		logger.info("Evaluating ... ");

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(featureFile));

		BufferedWriter out = null;
		if (outFile != null)
			out = new BufferedWriter(new FileWriter(outFile));

		for (int k = 0; k < data.length; k++) {

			SLInstance inst = data[k];
			// Features feats = data[k].getFeatures(in);
			Features feats = data[k].getFeatures();

			int nP = 5;
			SequencePrediction prediction = (SequencePrediction) predictor.decode(inst, feats, nP);

			String res = "";
			int nC = -1;
			int nT = 0;
			for (int j = 0; j < nP; j++) {
				String[] act_spans = ((SequenceLabel) inst.getLabel()).tags;
				SequenceLabel sl = prediction.getLabelByRank(j);
				if (sl == null)
					break;
				nT = 0;
				String[] pred_spans = sl.tags;

				String resT = "";
				int nCT = 0;
				for (int i = 1; i < pred_spans.length; i++) {
					String pred = pred_spans[i];
					String act = act_spans[i];
					if (pred.equals(act))
						nCT++;
					nT++;
					resT += pred_spans[i] + "\t";
				}
				if (nCT > nC) {
					nC = nCT;
					res = resT;
				}
				if (out != null)
					out.write(resT.trim() + "\n");
			}
			acc_corr += nC;
			acc_tot += nT;
		}

		in.close();
		if (out != null)
			out.close();

		DecimalFormat nf = new DecimalFormat();
		nf.setMaximumFractionDigits(5);

		logger.info("Total:");
		logger.info("\tTotal: " + acc_tot + ", Correct: " + acc_corr);
		logger.info("\tAccuracy: " + nf.format(((double) acc_corr / acc_tot)));

		logger.info("done.");
	}
}
