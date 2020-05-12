/* Copyright (C) 2006 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package struct.alg;

import java.io.IOException;
import java.util.logging.Logger;
import java.util.logging.FileHandler;
import java.util.logging.SimpleFormatter;
import java.util.logging.Level;
import java.util.logging.Handler;



import struct.types.Features;
import struct.types.SLEvaluator;
import struct.types.SLInstance;

/**
 * Learner to train the predictor with instances.
 *
 * @version 08/15/2006
 */
public class OnlineLearner implements StructuredLearner {
	private static Logger logger = Logger.getLogger(OnlineLearner.class.getName());
	private static FileHandler fh = null;
	private final boolean _averaging;

	private static void _initLogger(){
		try {
			fh=new FileHandler("src/main/java/examples/logs/logger.log", true);
		} catch (SecurityException | IOException e) {
			e.printStackTrace();
		}
		Logger l = Logger.getLogger("");
		SimpleFormatter sf = new SimpleFormatter();
		//sf.format="%4$: %5$ [%1$tc]%n";
		fh.setFormatter(sf);
		l.addHandler(fh);
		l.setLevel(Level.ALL);
	}

	private static void _terminateLogger(){
		for(Handler h:logger.getHandlers()){
			h.close();   //must call h.close or a .LCK file will remain.
		}
	}

	/**
	 * Averaging on by default.
	 */
	public OnlineLearner() {
		this._initLogger();
		this._averaging = true;

	}

	/**
	 *
	 * @param averaging
	 *            - Set averaging.
	 */
	public OnlineLearner(boolean averaging) {
		this._initLogger();
		this._averaging = averaging;
	}

	/**
	 * Trains the predictor with the given parameters.
	 *
	 * @throws IOException
	 */
	public void train(SLInstance[] training, OnlineUpdator updator, Predictor predictor, int numIters)
			throws IOException {

		for (int i = 0; i < numIters; i++) {
			logger.info("Training iteration " + i);

			//System.out.println("Training iteration " + i);

			long start = System.currentTimeMillis();

			for (int j = 0; j < training.length; j++) {
//				System.out.println("updating example " + j);
				SLInstance inst = training[j];
				Features feats = training[j].getFeatures();

				double avg_upd = numIters * training.length - (training.length * ((i + 1) - 1) + (j + 1)) + 1;

				updator.update(inst, feats, predictor, avg_upd);

			}

			if (this._averaging) {
				predictor.averageWeights(numIters * training.length);
			}

			long end = System.currentTimeMillis();
			logger.info(" took: " + ((float)(end - start)/1000) + "s");
		}
	}

	/**
	 * Trains the predictor with the given parameters and evaluates its
	 * performance. Always averages parameters.
	 *
	 * @throws IOException
	 */
	public void trainAndEvaluate(SLInstance[] training, SLInstance[] testing, OnlineUpdator update, Predictor predictor,
			int numIters, SLEvaluator eval) throws IOException {

		trainAndEvaluate(training, testing, update, predictor, numIters, eval, true);

	}

	/**
	 * Trains the predictor with the given parameters and evaluates its
	 * performance.
	 *
	 * @throws IOException
	 */
	public void trainAndEvaluate(SLInstance[] training, SLInstance[] testing, OnlineUpdator update, Predictor predictor,
			int numIters, SLEvaluator eval, boolean avgParams) throws IOException {

		for (int i = 0; i < numIters; i++) {
			logger.info(i + " ");
			logger.info("==========================");
			logger.info("Training iteration: " + i);
			logger.info("==========================");
			long start = System.currentTimeMillis();

			for (int j = 0; j < training.length; j++) {
				logger.info(".");

				SLInstance inst = training[j];
				Features feats = training[j].getFeatures();

				double avg_upd = numIters * training.length - (training.length * ((i + 1) - 1) + (j + 1)) + 1;

				update.update(inst, feats, predictor, avg_upd);
			}

			logger.info("");

			long end = System.currentTimeMillis();
			logger.info("Training took: " + (end - start));
			logger.info("Training");
			eval.evaluate(training, predictor);
			logger.info("Testing");
			eval.evaluate(testing, predictor);
		}
		logger.info("");

		if (avgParams) {
			logger.info("Averaging parameters...");
			predictor.averageWeights(numIters * training.length);
		}
		logger.info("==========================");
		logger.info("Final Performance.");
		logger.info("==========================");
		logger.info("Training");
		eval.evaluate(training, predictor);
		logger.info("Testing");
		eval.evaluate(testing, predictor);
	}
}
