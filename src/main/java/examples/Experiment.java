package examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.RoundingMode;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.text.DecimalFormat;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
import cc.mallet.pipe.TokenSequence2FeatureVectorSequence;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.types.*;
import struct.alg.*;
import struct.sequence.InstanceListDataManager;
import struct.sequence.LinearSequenceModel;

public class Experiment {

	private String nonEntity = "O";
	private double EPSILON = 1e-6;
	private enum Statistics {
		GOLD, PREDICTED, TRUE_POSITIVE;
	}
	private static Logger logger;
	private static FileHandler fh = null;
	private static final String OVERALL = "overall";
	HashMap<String, String> runParams = new HashMap<>();
	HashMap<String, String> runData = new HashMap<>();
	HashMap<String, Map<String, String>> trainStats = new HashMap<>();
	HashMap<String, Map<String, String>> testStats = new HashMap<>();
	HashMap<String, Map<String, String>> developStats = new HashMap<>();

	// sentence level statistics, combined for train, test and develop
	HashMap<String, Map<String, ArrayList<Double>>> sentenceLevelStats = new HashMap<>();

	private String logFilesPath;
	private String resultsFilePath;
	private String trainFilePath;
	private String testFilePath;
	private String developFilePath;
	private int numTrainIterations;
	private String updatorClass;
	private int completedFoldsCounter;
	private boolean createUnsupported;

	// MIRA specifics
	private int kMira;

	// SWVP and SWVM specifics
	private String maType;
	private boolean clusterMAsFlag;
	private int topK;
	private String jjType;

	private String gammaCalculationMethod;
    private double gammaCalculationBeta;
    private String gammaCalculationObjectiveType;

	// feature map
	private HashMap<String, Boolean> featureMap = new HashMap<>();


	Experiment(HashMap<String, String> runParams, HashMap<String, Boolean> featureMap, Logger logger) {

		// validate the existence of required directories
		validateDirs();

		this.logger = logger;

		logger.info("parsing experiment parameters...");
		try {

			// -------- DEPRECATED PARAMETERS BEGIN HERE -------- //

			// determine if you cluster Mixed Assignment
			// TODO: make clusterMasFlag=True work
			runParams.put("clusterMAsFlag", "false");

			// -------- DEPRECATED PARAMETERS END HERE -------- //

			// run parameters
			this.runParams = runParams;

			// createUnsupported
			if (runParams.get("createUnsupported") != null){
				String createUns = runParams.get("createUnsupported");
				if (createUns.equals("true")){
					createUnsupported = true;
				} else if (createUns.equals("false")){
					createUnsupported = false;
				} else { throw new IllegalArgumentException("must specify true or false for createUnsupported (boolean)"); }
			} else{ throw new IllegalArgumentException("must specify a value for createUnsupported (boolean)"); }

			// completedFoldsCounter
			// when running from CrossValidation we want to know in which fold we are in
			if (runParams.get("completedFoldsCounter") != null){
				completedFoldsCounter = Integer.valueOf(runParams.get("completedFoldsCounter"));
			} else{
				completedFoldsCounter = 0;
			}

			// log
			if (runParams.get("logFilesPath") != null) {
				logFilesPath = runParams.get("logFilesPath");
			} else { throw new IllegalArgumentException("must specify log file path"); }

			// data
			if (runParams.get("trainFilePath") != null) {
				trainFilePath = runParams.get("trainFilePath");
			} else { throw new IllegalArgumentException("must specify train file path"); }
			if (runParams.get("testFilePath") != null) {
				testFilePath = runParams.get("testFilePath");
			} else { throw new IllegalArgumentException("must specify test file path"); }
			if (runParams.get("developFilePath") != null) {
				developFilePath = runParams.get("developFilePath");
			} else { throw new IllegalArgumentException("must specify develop file path"); }

			// results
			if (runParams.get("resultsFilePath") != null) {
				resultsFilePath = runParams.get("resultsFilePath");
			} else { throw new IllegalArgumentException("must specify output file path"); }

			// number of training iterations
			if (runParams.get("numTrainIterations") != null)
			{
				int numIters = Integer.parseInt(runParams.get("numTrainIterations"));
				if(numIters > 0) {
					numTrainIterations = numIters;
				} else{ throw new IllegalArgumentException("number of iterations must be a positive integer"); }
			} else {
				logger.log(Level.WARNING, "number of iterations not specified - using default value of 15 iterations");
				numTrainIterations = 15;
			}

			// updator class
			if (runParams.get("updatorClass") != null) {
				String updatorCls = runParams.get("updatorClass");
				if (updatorCls.equals("PerceptronUpdator")){ updatorClass = "PerceptronUpdator"; }
				else if (updatorCls.equals("KBestMiraUpdator")){
					updatorClass = "KBestMiraUpdator";
					if (runParams.get("kMira") != null && Integer.parseInt(runParams.get("kMira")) > 0){
						kMira = Integer.parseInt(runParams.get("kMira"));
					}
					else{ throw new IllegalArgumentException("must specify positive Integer K value for K-mira updator"); }
				}
				else if (updatorCls.equals("SWVPUpdator")) {
					updatorClass = "SWVPUpdator";

					// -------- DEPRECATED PARAMETERS BEGIN HERE -------- //
					if (runParams.get("clusterMAsFlag") != null) {
						String clsMAflag = runParams.get("clusterMAsFlag");

						if (clsMAflag.equals("true") || (clsMAflag.equals("false"))){
							clusterMAsFlag = Boolean.parseBoolean(clsMAflag);
						} else{ throw new IllegalArgumentException("must specify a legal boolean value for clusterMAsFlag (true/false) for JJ updator"); }
					} else { throw new IllegalArgumentException("clusterMAsFlag value not specified (true/false) for SWVP updator"); }
					// -------- DEPRECATED PARAMETERS END HERE -------- //

					if (runParams.get("topK") != null) {
						int tpK = Integer.valueOf(runParams.get("topK"));
						if (tpK >= 1 && tpK <= 20) {
							topK = tpK;
						} else {
							throw new IllegalArgumentException("must specify a value in the range [1,20] for topK"); }
					} else { throw new IllegalArgumentException("must specify a value top topK"); }

					if (runParams.get("maType") != null && (runParams.get("maType").equals("passive")) ||
							(runParams.get("maType").equals("aggressive")) || (runParams.get("maType").equals("all"))
							) {
						maType = runParams.get("maType");
					} else { throw new IllegalArgumentException("must specify legal MA type (passive/aggressive/all) for SWVP updator"); }

					if (runParams.get("jjType") != null) {
						String jjT = runParams.get("jjType");
						if (jjT.equals("single") || jjT.equals("double") || jjT.equals("triple")){
							this.jjType = jjT;
						} else { throw new IllegalArgumentException("must specify 'single' or 'double' or 'triple' for jjType"); }
					} else { throw new IllegalArgumentException("must specify a value for jjType"); }

					if (runParams.get("gammaCalculationMethod") != null) {
						String gammaClcMethod = runParams.get("gammaCalculationMethod");
						if (gammaClcMethod.equals("uniform") ||
								gammaClcMethod.equals("wm") || gammaClcMethod.equals("wmr") ||
								gammaClcMethod.equals("softmax") || gammaClcMethod.equals("softmin") ||
								gammaClcMethod.equals("one") || gammaClcMethod.equals("opt")) {
							gammaCalculationMethod = gammaClcMethod;
						} else {
							throw new IllegalArgumentException("gammaCalculationMethod can only be uniform or wm or wmr " +
									"or softmax or softmin or one or opt"); }
					} else { throw new IllegalArgumentException("must specify gammaCalculationMethod"); }

					if (runParams.get("gammaCalculationBeta") != null) {
						double gammaClcBeta = Double.valueOf(runParams.get("gammaCalculationBeta"));
						if (gammaClcBeta > 0.1 - EPSILON && gammaClcBeta < 10 + EPSILON) {
							gammaCalculationBeta = gammaClcBeta;
						} else {
							throw new IllegalArgumentException("gammaCalculationBeta must be in the range [0.1,10]"); }
					} else { throw new IllegalArgumentException("must specify gammaCalculationBeta"); }

                    if (runParams.get("gammaCalculationObjectiveType") != null){
                        String objType = runParams.get("gammaCalculationObjectiveType");
                        if (objType.equals("MINIMIZE") || objType.equals("MAXIMIZE")){
                            gammaCalculationObjectiveType = objType;
                        } else { throw new IllegalArgumentException("must specify MINIMIZE or MAXIMIZE for gammaCalculationObjectiveType"); }
                    } else{ throw new IllegalArgumentException("must specify legal objective type (MAXIMIZE/MINIMIZE) for SWVP updator"); }

				}

				else if (updatorCls.equals("SWVMUpdator")) {
					updatorClass = "SWVMUpdator";

					// -------- DEPRECATED PARAMETERS BEGIN HERE -------- //
					if (runParams.get("clusterMAsFlag") != null) {
						String clsMAflag = runParams.get("clusterMAsFlag");

						if (clsMAflag.equals("true") || (clsMAflag.equals("false"))){
							clusterMAsFlag = Boolean.parseBoolean(clsMAflag);
						} else{ throw new IllegalArgumentException("must specify a legal boolean value for clusterMAsFlag (true/false) for SWVM updator"); }
					} else { throw new IllegalArgumentException("clusterMAsFlag value not specified (true/false) for SWVM updator"); }
					// -------- DEPRECATED PARAMETERS END HERE -------- //

					if (runParams.get("topK") != null) {
						int tpK = Integer.valueOf(runParams.get("topK"));
						if (tpK >= 1 && tpK <= 20) {
							topK = tpK;
						} else {
							throw new IllegalArgumentException("must specify a value in the range [1,20] for topK"); }
					} else { throw new IllegalArgumentException("must specify a value top topK"); }

					if (runParams.get("maType") != null && (runParams.get("maType").equals("passive")) ||
							(runParams.get("maType").equals("aggressive")) || (runParams.get("maType").equals("all"))
							) {
						maType = runParams.get("maType");
					} else { throw new IllegalArgumentException("must specify legal MA type (passive/aggressive/all) for SWVM updator"); }

					if (runParams.get("jjType") != null) {
						String jjT = runParams.get("jjType");
						if (jjT.equals("single") || jjT.equals("double") || jjT.equals("triple")){
							this.jjType = jjT;
						} else { throw new IllegalArgumentException("must specify 'single' or 'double' or 'triple' for jjType"); }
					} else { throw new IllegalArgumentException("must specify a value for jjType"); }

					if (runParams.get("gammaCalculationMethod") != null) {
						String gammaClcMethod = runParams.get("gammaCalculationMethod");
						if (gammaClcMethod.equals("uniform") ||
								gammaClcMethod.equals("wm") || gammaClcMethod.equals("wmr") ||
								gammaClcMethod.equals("softmax") || gammaClcMethod.equals("softmin") ||
								gammaClcMethod.equals("one") || gammaClcMethod.equals("opt")) {
							gammaCalculationMethod = gammaClcMethod;
						} else {
							throw new IllegalArgumentException("gammaCalculationMethod can only be uniform or wm or wmr " +
									"or softmax or softmin or one or opt"); }
					} else { throw new IllegalArgumentException("must specify gammaCalculationMethod"); }

					if (runParams.get("gammaCalculationBeta") != null) {
						double gammaClcBeta = Double.valueOf(runParams.get("gammaCalculationBeta"));
						if (gammaClcBeta > 0.1 - EPSILON && gammaClcBeta < 10 + EPSILON) {
							gammaCalculationBeta = gammaClcBeta;
						} else {
							throw new IllegalArgumentException("gammaCalculationBeta must be in the range [0.1,10]"); }
					} else { throw new IllegalArgumentException("must specify gammaCalculationBeta"); }

                    if (runParams.get("gammaCalculationObjectiveType") != null){
                        String objType = runParams.get("gammaCalculationObjectiveType");
                        if (objType.equals("MINIMIZE") || objType.equals("MAXIMIZE")){
                            gammaCalculationObjectiveType = objType;
                        } else { throw new IllegalArgumentException("must specify MINIMIZE or MAXIMIZE for gammaCalculationObjectiveType"); }
                    } else{ throw new IllegalArgumentException("must specify legal objective type (MAXIMIZE/MINIMIZE) for SWVM updator"); }

				}
				else{ throw new IllegalArgumentException("illegal updator class"); }
			} else { throw new IllegalArgumentException("must specify an updator class"); }


			// feature map
			if (featureMap != null)
			{
				this.featureMap = featureMap;
			} else{ throw new IllegalArgumentException("must specify a feature map object"); }

			}catch (Exception e){
			logger.log(Level.SEVERE, e.getMessage());
			System.exit(1);
		}
	}

	public void runExperiment() throws IOException {

		// add experiment beginning datetime to runData
		SimpleDateFormat beganExperimentDateFormat = new SimpleDateFormat("HH:mm:ss");
		String beganExperimentDatetime = beganExperimentDateFormat.format(new Date());
		runData.put("began experiment datetime", beganExperimentDatetime);
		logger.info("began Experiment " + String.valueOf(completedFoldsCounter+1) + "\n");

		// print run parameters
		printRunParams();

		// print feature map
		printFeatureMap();

		// add train/test pathname to run data
		runData.put("train file pathname", trainFilePath);
		runData.put("test file pathname", testFilePath);
		runData.put("develop file pathname", developFilePath);
		logger.info("train file pathname: " + trainFilePath);
		logger.info("test file pathname: " + testFilePath);
		logger.info("develop file pathname: " + developFilePath);

		// add training pipline beginning datetime to runData
		SimpleDateFormat beganTrainingPipelineDateFormat = new SimpleDateFormat("HH:mm:ss");
		String beganTrainingPipelineDatetime = beganTrainingPipelineDateFormat.format(new Date());
		runData.put("began training pipeline datetime", beganTrainingPipelineDatetime);
		logger.info("began training pipeline");

		// make a mallet pipe to create features from the input file
		Pipe p = new SerialPipes(
				new Pipe[] { new SimpleTaggerSentence2TokenSequence(), new TokenSequence2FeatureVectorSequence() });
		// make a list of instances for the training data
		InstanceList trainList = new InstanceList(p);
		// trainList.add( new LineGroupIterator(new FileReader(new
		// File("data/conll.train")), Pattern.compile("^\\s*$"), true));
		trainList.addThruPipe(new LineGroupIterator(new FileReader(new File(trainFilePath)),
				Pattern.compile("^\\s*$"), true));

		// validate the existence of the nonEntity in the data set
		if (!trainList.getTargetAlphabet().contains(nonEntity)){
			throw new AssertionError("Supporting only data sets containing non-entity - " + String.valueOf(nonEntity));
		}

        // validate data structure is looking as it should
		// validateListStructure(trainList);

        // add train size to run data
        String trainListSize = String.valueOf(trainList.size());
        runData.put("train size", trainListSize);
        logger.info("No. of training instances = " + trainListSize);

		// create a manager object to convert between mallet's feature vector
		// representation and that used by StrunctLearn
		InstanceListDataManager manager = new InstanceListDataManager(trainList, featureMap);
		// create the alphabets, including features that are not supported by
		// the observed data

		manager.createAlphabets(createUnsupported);

		manager.setInstanceAlphabets();
		manager.dataAlphabet.stopGrowth();

		// create a linear sequence model for the given
		LinearSequenceModel lsm = new LinearSequenceModel(manager);

		OnlineUpdator updator = null;

		if (updatorClass.equals("PerceptronUpdator")){
			updator = new PerceptronUpdator();
		}
		else if (updatorClass.equals("KBestMiraUpdator")){
			updator = new KBestMiraUpdator(kMira);
		}
		else if (updatorClass.equals("SWVPUpdator")){
			updator = new SWVPUpdator(clusterMAsFlag, topK, maType, jjType,
                    gammaCalculationMethod, gammaCalculationBeta, gammaCalculationObjectiveType);
		}
		else if (updatorClass.equals("SWVMUpdator")){
			updator = new SWVMUpdator(clusterMAsFlag, topK, maType, jjType,
                    gammaCalculationMethod, gammaCalculationBeta, gammaCalculationObjectiveType);
		}

		// add updator class to runData
		String updatorClass = updator.getClass().toString();
		runData.put("updator class", updatorClass);
		logger.info("updator class: " + updatorClass);

		/////////// TRAIN //////////////////////////////////////////////////////////////////
		// add training beginning datetime to runData
		SimpleDateFormat beganTrainingDateFormat = new SimpleDateFormat("HH:mm:ss");
		String beganTrainingDatetime = beganTrainingDateFormat.format(new Date());
		runData.put("began training datetime", beganTrainingDatetime);
		logger.info("began training");

		String exceptionCauseStr;
		try {
			lsm.train(trainList, numTrainIterations, updator);
		} catch (NullPointerException e){
			exceptionCauseStr = e.getCause().toString();
			logger.log(Level.SEVERE, exceptionCauseStr);
			System.exit(1);
		}
		catch (Exception e){
			exceptionCauseStr = e.getCause().toString();
			logger.log(Level.SEVERE, exceptionCauseStr);
			System.exit(1);
		}

		// add number of iterations to runData
		runData.put("number of training iterations", String.valueOf(numTrainIterations));
		logger.info("number of training iterations: " + String.valueOf(numTrainIterations));

		// add training finish datetime to runData
		SimpleDateFormat finishedTrainingDateFormat = new SimpleDateFormat("HH:mm:ss");
		String finishedTrainingDatetime = finishedTrainingDateFormat.format(new Date());
		runData.put("finished training datetime", finishedTrainingDatetime);
		logger.info("finished training");

		// evaluate on the training data
		logger.info("Training ");

		// calc train metrics
		trainStats = calcMetrics(trainList, lsm, "train");
		// HashMap<String, Map<String, String>> trainStats = calcMetrics(trainList, lsm);

		// calc train accuracy and add it to trainStats
		String trainAccuracy = String.valueOf(calcAccuracy(trainList, lsm));
		HashMap<String, String> tempTrainAccMap = new HashMap<>();
		tempTrainAccMap.put("overall", trainAccuracy);
		trainStats.put("Accuracy", tempTrainAccMap);
		logger.info("Train Accuracy: " + trainAccuracy);
		////////////////////////////////////////////////it//////////////////////////////

		/////////// TEST //////////////////////////////////////////////////////////////////
		// add test pipeline beginning datetime to runData
		SimpleDateFormat beganTestingPipelineDateFormat = new SimpleDateFormat("HH:mm:ss");
		String beganTestingPipelineDatetime = beganTestingPipelineDateFormat.format(new Date());
		runData.put("began testing pipeline datetime", beganTestingPipelineDatetime);
		logger.info("began testing pipeline");


		// make a list of instance for the testing data
		InstanceList testList = new InstanceList(p);
		// testList.add(new LineGroupIterator(new FileReader(new
		// File("data/conll.test")),Pattern.compile("^\\s*$"), true));
		// testList.add(new LineGroupIterator(new FileReader(new
		// File("data/genia/genia_develop.txt")),Pattern.compile("^\\s*$"),
		// true));
		testList.addThruPipe(new LineGroupIterator(new FileReader(new File(testFilePath)),
				Pattern.compile("^\\s*$"), true));

		// add test size to run data
		String testListSize = String.valueOf(testList.size());
		runData.put("test size", testListSize);
		logger.info("No. of testing instances = " + testListSize);

		// add test pipeline finish datetime to runData
		SimpleDateFormat finishedTestingPipelineDateFormat = new SimpleDateFormat("HH:mm:ss");
		String finishedTestingPipelineDatetime = finishedTestingPipelineDateFormat.format(new Date());
		runData.put("finished testing pipeline datetime", finishedTestingPipelineDatetime);
		logger.info("finished testing pipeline");

		// evaluate on the testing data
		logger.info("Testing ");

		// calc test metrics
		testStats = calcMetrics(testList, lsm, "test");

		//HashMap<String, Map<String, String>> testStats = calcMetrics(testList, lsm);

		// calc test accuracy and add it to trainStats
		String testAccuracy = String.valueOf(calcAccuracy(testList, lsm));
		HashMap<String, String> tempTestAccMap = new HashMap<>();
		tempTestAccMap.put("overall", testAccuracy);
		testStats.put("Accuracy", tempTestAccMap);
		logger.info("Test Accuracy: " + testAccuracy);
		//////////////////////////////////////////////////////////////////////////////

		/////////// DEVELOP //////////////////////////////////////////////////////////////////
		// add develop pipeline beginning datetime to runData
		SimpleDateFormat beganDevelopPipelineDateFormat = new SimpleDateFormat("HH:mm:ss");
		String beganDevelopPipelineDatetime = beganDevelopPipelineDateFormat.format(new Date());
		runData.put("began develop pipeline datetime", beganDevelopPipelineDatetime);
		logger.info("began develop pipeline");


		// make a list of instance for the develop data
		InstanceList developList = new InstanceList(p);
		// testList.add(new LineGroupIterator(new FileReader(new
		// File("data/conll.develop")),Pattern.compile("^\\s*$"), true));
		// developList.add(new LineGroupIterator(new FileReader(new
		// File("data/genia/genia_develop.txt")),Pattern.compile("^\\s*$"),
		// true));
		developList.addThruPipe(new LineGroupIterator(new FileReader(new File(developFilePath)),
				Pattern.compile("^\\s*$"), true));

		// add develop size to run data
		String developListSize = String.valueOf(developList.size());
		runData.put("develop size", developListSize);
		logger.info("No. of develop instances = " + developListSize);

		// add develop pipeline finish datetime to runData
		SimpleDateFormat finishedDevelopPipelineDateFormat = new SimpleDateFormat("HH:mm:ss");
		String finishedDevelopPipelineDatetime = finishedDevelopPipelineDateFormat.format(new Date());
		runData.put("finished develop pipeline datetime", finishedDevelopPipelineDatetime);
		logger.info("finished develop pipeline");

		// evaluate on the develop data
		logger.info("Develop ");

		// calc develop metrics
		developStats = calcMetrics(developList, lsm, "develop");
		//HashMap<String, Map<String, String>> testStats = calcMetrics(develop, lsm);

		// calc develop accuracy and add it to trainStats
		String developAccuracy = String.valueOf(calcAccuracy(developList, lsm));
		HashMap<String, String> tempDevelopAccMap = new HashMap<>();
		tempDevelopAccMap.put("overall", developAccuracy);
		developStats.put("Accuracy", tempDevelopAccMap);
		logger.info("Develop Accuracy: " + developAccuracy);
		//////////////////////////////////////////////////////////////////////////////


		String updatorClassStr = updator.getClass().toString().replace("class struct.alg.", "");
		runData.put("updator class", updatorClassStr);

		// add experiment finish datetime to runData
		SimpleDateFormat finishExperimentDateFormat = new SimpleDateFormat("HH:mm:ss");
		String finishExperimentDatetime = finishExperimentDateFormat.format(new Date());
		runData.put("finish experiment datetime", finishExperimentDatetime);
		logger.info("finished Experiment " + String.valueOf(completedFoldsCounter+1) + "\n");

		// add experiment duration to runData
		long secondsDiff = calcExperimentDurationFromRunData(runData);
		long minutesDiff = secondsDiff / 60;

		String experimentDuration = String.valueOf(minutesDiff) + "m (" + String.valueOf(secondsDiff) + "s)";
		runData.put("experiment duration", experimentDuration);

		// write results to file..
		writeResultsToFile();

	}

	private void validateListStructure(InstanceList instList){
		int instSize = instList.size();

		for (int i=0; i<instSize; i++){

			LabelSequence currTarget = (LabelSequence) instList.get(i).getTarget();
			Sequence currSequence = (Sequence) (instList.get(i).getData());

			int currTargetSize = currTarget.size();
			int currSequenceSize = currSequence.size();

			ArrayList<Integer> problemIndices = new ArrayList<>();
			if (currTargetSize != currSequenceSize){
				problemIndices.add(i);
			}
		}
	}

	private void writeResultsToFile(){

		SimpleDateFormat dateFormat = new SimpleDateFormat("yyMMdd-HH-mm-ss");
		String date = dateFormat.format(new Date());

		// long filename
		String fileNameInfo = date + "__numiter=" + runData.get("number of training iterations") +
				"__testAccuracy=" + String.format("%.3f", Double.valueOf(testStats.get("Accuracy").get("overall"))) + "__testF1=" + String.format("%.3f", Double.valueOf(testStats.get("F1").get("overall"))) +
				"__developAccuracy=" + String.format("%.3f", Double.valueOf(developStats.get("Accuracy").get("overall"))) + "__developF1=" + String.format("%.3f", Double.valueOf(developStats.get("F1").get("overall")));


		// short filename
//		String fileNameInfo = date + "__" + updatorClassStr;

		String fileName = resultsFilePath + String.valueOf(completedFoldsCounter+1) + "__" + fileNameInfo + ".txt";



		try{
			PrintWriter writer = new PrintWriter(fileName, "UTF-8");
			writer.println("---- Updator class ----");
			writer.println(runData.get("updator class"));
			writer.println("\n");

			writer.println("---- Bottom lines ----");
			writer.println("test Accuracy: " + testStats.get("Accuracy").get("overall"));
			writer.println("test F1: " + testStats.get("F1").get("overall"));

			writer.println("develop Accuracy: " + developStats.get("Accuracy").get("overall"));
			writer.println("develop F1: " + developStats.get("F1").get("overall"));

			writer.println("train Accuracy: " + trainStats.get("Accuracy").get("overall"));
			writer.println("train F1: " + trainStats.get("F1").get("overall"));
			writer.println("\n");

			writer.println("---- Run Data ----");
			for (Map.Entry<String, String> entry : runData.entrySet()){
				String key = entry.getKey();
				String value = entry.getValue();
				writer.println(key + ": " + value);
			}
			writer.println("\n");

			writer.println("---- Run Parameters ----");
			for (Map.Entry<String, String> entry : runParams.entrySet()){
				String key = entry.getKey();
				String value = entry.getValue();
				writer.println(key + ": " + value);
			}
			writer.println("\n");

			writer.println("---- Feature Map ----");
			for (Map.Entry<String, Boolean> entry : featureMap.entrySet()){
				String key = entry.getKey();
				String value = String.valueOf(entry.getValue());
				writer.println(key + ": " + value);
			}
			writer.println("\n");

			writer.println("---- Train Begin ----");
			//writer.println("Accuracy: " + trainStats.get("Accuracy").get("overall"));

			for (Map.Entry<String, Map<String, String>> entry : trainStats.entrySet()) {
				String metric = entry.getKey();
				Map<String, String> val = entry.getValue();
				writer.println(metric + ": " + val.get("overall"));
			}
			writer.println("\n");

			for (Map.Entry<String, Map<String, String>> entry : trainStats.entrySet()){
				String metric = entry.getKey();
				Map<String, String> val = entry.getValue();

				writer.println("# " + metric + " per entity type # ");
				for (String entityType : val.keySet()){
					writer.print(entityType + ": " + val.get(entityType) + "\t");
				}
				writer.println("\n");

			}
			writer.println("---- Train End ----");
			writer.println("\n");
			writer.println("---- Test Begin ----");
			//writer.println("Accuracy: " + testStats.get("Accuracy").get("overall"));

			for (Map.Entry<String, Map<String, String>> entry : testStats.entrySet()) {
				String metric = entry.getKey();
				Map<String, String> val = entry.getValue();
				writer.println(metric + ": " + val.get("overall"));
			}
			writer.println("\n");

			for (Map.Entry<String, Map<String, String>> entry : testStats.entrySet()){
				String metric = entry.getKey();
				Map<String, String> val = entry.getValue();

				writer.println("# " + metric + " per entity type # ");
				for (String entityType : val.keySet()){
					writer.print(entityType + ": " + val.get(entityType) + "\t");
				}
				writer.println("\n");

			}
			writer.println("---- Test End ----");
			writer.println("\n");
			writer.println("---- Develop Begin ----");
			//writer.println("Accuracy: " + developStats.get("Accuracy").get("overall"));

			for (Map.Entry<String, Map<String, String>> entry : developStats.entrySet()) {
				String metric = entry.getKey();
				Map<String, String> val = entry.getValue();
				writer.println(metric + ": " + val.get("overall"));
			}
			writer.println("\n");

			for (Map.Entry<String, Map<String, String>> entry : developStats.entrySet()){
				String metric = entry.getKey();
				Map<String, String> val = entry.getValue();

				writer.println("# " + metric + " per entity type # ");
				for (String entityType : val.keySet()){
					writer.print(entityType + ": " + val.get(entityType) + "\t");
				}
				writer.println("\n");

			}
			writer.println("---- Develop End ----");
			writer.close();
		} catch (IOException e) {
			logger.severe(e.getMessage());
		}
	}

	private HashMap<String, Map<String, String>> calcMetrics(InstanceList testList, LinearSequenceModel lsm, String dataType) {

		Alphabet tags = testList.getTargetAlphabet();
		String labels[] = new String[tags.size()];
		tags.toArray(labels);

		Map<Statistics, Map<String, Long>> stats = new HashMap<>();

		// calculates labels histogram for gold, predicted and true positive
		// statistics count number of occurrences of each label and the number
		// of occurrences of non entity for each label

		calcStatistics(testList, lsm, stats, nonEntity);
		String statsStr = getStatisticsString(stats);
		logger.info(statsStr);
		HashMap<String, Map<String, String>> resMap = getStatisticsMap(stats);

		calcSentenceLevelStatistics(testList, lsm, nonEntity, dataType);

		return resMap;
	}

	/**
	 * calculates sentence level (instanace level) statistics and saves it to sentenceLevelStats class member
	 * @param instList
	 * @param lsm
	 * @param nonEntity
	 * @param dataType
	 *
	 * @return nothing
	 */
	void calcSentenceLevelStatistics(InstanceList instList, LinearSequenceModel lsm, String nonEntity, String dataType){

		logger.fine("began calcSentenceLevelStatistics() for " + String.valueOf(dataType) + "data");

		// data set level lists
		ArrayList<Double> dataAccuracy = new ArrayList();
		ArrayList<Double> dataPrecision = new ArrayList();
		ArrayList<Double> dataRecall = new ArrayList();
		ArrayList<Double> dataF1 = new ArrayList();

		// for each test instance (sentence)
		for (int i = 0; i < instList.size(); i++) {
			// real values
			LabelSequence realSeq = (LabelSequence) instList.get(i).getTarget();
			// predicted values
			Sequence predictedSeq = lsm.transduce((Sequence) (instList.get(i).getData()));

			// instance level statistics
			double instAccuracy = 0;
			double instPrecision = 0;
			double instRecall = 0;
			double instF1 = 0;

			double tp = 0;
			double tn = 0;
			double fp = 0;
			double fn = 0;

			int instSize = realSeq.size();

			// calculate metrics for each instance (sentence)
			for (int j = 0; j < instSize; j++) {
				String trueLabel = realSeq.getLabelAtPosition(j).toString();
				String predLabel = (String) predictedSeq.get(j);

				 // "DNA" == "DNA true and pred are same
				if (trueLabel.equals(predLabel)){
					instAccuracy++;

					// true label is NonEntity -  increase true negative
					if (trueLabel.equals(nonEntity)){
						tn++;
					} else { // true label is Entity - increase true positive
						tp++;
					}
				} else { // "DNA" != "RNA" true and pred differ
					// true label is NonEntity -  increase false positive
					if (trueLabel.equals(nonEntity)){
						fp++;
					} else { // true label is Entity - increase false negative
						fn++;
					}
				}
			}

			// calculate statistics for the instance (sentence)
			try {
				instAccuracy = instAccuracy / (double) instSize;
				if (tp + fp == 0){
					instPrecision = 0;
					instF1 = 0;
				} else if (tp + fn == 0){
					instRecall = 0;
					instF1 = 0;
				} else {
					instPrecision =  tp / (tp + fp);
					instRecall =  tp / (tp + fn);
					if (instPrecision + instRecall == 0){
						instF1 = 0;
					} else {
						instF1 = 2 * (instPrecision * instRecall) / (instPrecision + instRecall);
					}
				}
			} catch (Exception e){
				logger.severe(e.getMessage());
			}

			// add to data lists
			dataAccuracy.add(instAccuracy);
			dataPrecision.add(instPrecision);
			dataRecall.add(instRecall);
			dataF1.add(instF1);
		}

		// add to experiment HashMap
		Map<String, ArrayList<Double>> currMap = new HashMap<>();
		currMap.put("Accuracy", dataAccuracy);
		currMap.put("Precision", dataPrecision);
		currMap.put("Recall", dataRecall);
		currMap.put("F1", dataF1);

		sentenceLevelStats.put(dataType, currMap);
	}

	/**
	 * returns a HashMap including the precision, recall and f1 measure for all
	 * the labels and for the overall entities
	 */
	private HashMap<String, Map<String, String>> getStatisticsMap(Map<Statistics, Map<String, Long>> stats) {
		Set<String> labels = new HashSet<>();
		for (Statistics stat : stats.keySet()) {
			labels.addAll(stats.get(stat).keySet());
		}
		HashMap<String, Map<String, String>> resultMap = new HashMap<>();

		Map<String, Long> goldStats = stats.get(Statistics.GOLD), predStats = stats.get(Statistics.PREDICTED),
				tpStats = stats.get(Statistics.TRUE_POSITIVE);

		HashMap<String, String> precMap = new HashMap<>();
		HashMap<String, String> recMap = new HashMap<>();
		HashMap<String, String> f1Map = new HashMap<>();

		for (String l : labels){

			double prec, rec, f1;

			// predStats.get(l) is Long, prec is Double
			prec = predStats.get(l) == 0L ? 0.0 : (double) tpStats.get(l) / predStats.get(l);
			// rec is Double
			rec = (double) tpStats.get(l) / goldStats.get(l);
			f1 = (prec == 0 && rec == 0) ? 0.0 : 2 * (prec * rec) / (prec + rec);

			precMap.put(l, String.valueOf(prec));
			recMap.put(l, String.valueOf(rec));
			f1Map.put(l, String.valueOf(f1));
		}
		resultMap.put("Precision", precMap);
		resultMap.put("Recall", recMap);
		resultMap.put("F1", f1Map);

		return resultMap;
	}

	/**
	 * returns a string including the precision, recall and f1 measure for all
	 * the labels and for the overall entities
	 */
	private String getStatisticsString(Map<Statistics, Map<String, Long>> stats) {
		Set<String> labels = new HashSet<>();
		for (Statistics stat : stats.keySet()) {
			labels.addAll(stats.get(stat).keySet());
		}

		Map<String, Long> goldStats = stats.get(Statistics.GOLD), predStats = stats.get(Statistics.PREDICTED),
				tpStats = stats.get(Statistics.TRUE_POSITIVE);

		StringBuilder sb = new StringBuilder("Statistics:").append('\n');
		DecimalFormat df = new DecimalFormat("#.#####");
		df.setRoundingMode(RoundingMode.CEILING);

		for (String l : labels) {

			double prec, rec, f1;
			if (predStats.get(l) == 0L){
				prec = 0.0;
			} else {
				prec = ((double) tpStats.get(l)) / predStats.get(l);
			}

			rec = ((double) tpStats.get(l)) / goldStats.get(l);

			if (prec == 0 && rec == 0) {
				f1 = 0.0;
			} else{
				f1 = 2 * (prec * rec) / (prec + rec);
			}
			sb.append(l).append(": ");
			sb.append("Prec.:").append(df.format(prec)).append('\t');
			sb.append("Rec.:").append(df.format(rec)).append('\t');
			sb.append("F1.:").append(df.format(f1)).append('\n');
		}

		return sb.toString();
	}

	/**
	 * calculates labels histogram for gold, predicted and true positive
	 * statistics count number of occurrences of each label and the number of
	 * occurrences of non entity for each label
	 */
	private void calcStatistics(InstanceList testList, LinearSequenceModel lsm,
								Map<Statistics, Map<String, Long>> stats, String nonEntity) {

		for (Statistics stat : Statistics.values()) {
			stats.put(stat, new HashMap<>());
		}

		Map<String, Long> goldStats = stats.get(Statistics.GOLD), predStats = stats.get(Statistics.PREDICTED),
				tpStats = stats.get(Statistics.TRUE_POSITIVE);

		// for each test instance
		for (int i = 0; i < testList.size(); i++) {
			LabelSequence trueSeq = (LabelSequence) testList.get(i).getTarget();
			Sequence predictedSeq = lsm.transduce((Sequence) (testList.get(i).getData()));

			// calculate metrics for each token in the test instance
			for (int j = 0; j < trueSeq.size(); j++) {
				String trueLab = trueSeq.getLabelAtPosition(j).toString();
				String predLab = (String) predictedSeq.get(j);

				incOrCreate(goldStats, trueLab);
				incOrCreate(predStats, predLab);
				if (!trueLab.equals(nonEntity))
					incOrCreate(goldStats, OVERALL);
				if (!predLab.equals(nonEntity))
					incOrCreate(predStats, OVERALL);
				if (predLab.equals(trueLab)) {
					incOrCreate(tpStats, trueLab);
					if (!trueLab.equals(nonEntity))
						incOrCreate(tpStats, OVERALL);
				}
			}

			// deal with the case where there are tags in the gold label which were not predicted.
			for (String key : goldStats.keySet()) {
				Long predValue = predStats.get(key);
				Long tpValue = tpStats.get(key);

				if (predValue == null) {
					predStats.put(key, 0L);
				}
				if (tpValue == null) {
					tpStats.put(key, 0L);
				}
			}
            // deal with the case where there are predicted tags which were not in the gold label.
            for (String key : predStats.keySet()) {
                Long goldValue = goldStats.get(key);
                Long tpValue = tpStats.get(key);

                if (goldValue == null) {
                    goldStats.put(key, 0L);
                }
                if (tpValue == null) {
                    tpStats.put(key, 0L);
                }
            }

//			if ((i % 500 == 0)){
//				logger.fine("evaluating instance num. " + String.valueOf(i));
//			}

		}
	}

	private void incOrCreate(final Map<String, Long> goldStats, String k) {
		if (!goldStats.containsKey(k))
			goldStats.put(k, 0l);
		goldStats.put(k, goldStats.get(k) + 1);
	}

	/**
	 * custom accuracy testing (implements a simple calculation of token
	 * accuracy).
	 */
	private double calcAccuracy(InstanceList testList, LinearSequenceModel lsm) throws FileNotFoundException {
		int totalLabels = 0;
		int correctLabels = 0;

		for (int i = 0; i < testList.size(); i++) {
			// get the true labeling
			LabelSequence trueSeq = (LabelSequence) testList.get(i).getTarget();
			totalLabels += trueSeq.size();

			// get the predicted labeling
			Sequence input = (Sequence) (testList.get(i).getData());
			Sequence predictedSeq = lsm.transduce(input);

			// for each label, check for correctness
			for (int j = 0; j < trueSeq.size(); j++) {
				String predLabel = (String) predictedSeq.get(j);
				if (predLabel.equals(trueSeq.getLabelAtPosition(j).toString()))
					correctLabels++;
			}
		}
		double accuracy = ((double) correctLabels) / totalLabels;
		return accuracy;
	}

	HashMap<String, String> getRunData() { return runData; }
	HashMap<String, Map<String, String>> getTrainStats() { return trainStats; }
	HashMap<String, Map<String, String>> getTestStats() { return testStats; }
	HashMap<String, Map<String, String>> getDevelopStats() { return developStats; }
	HashMap<String, Map<String, ArrayList<Double>>> getSentenceLevelStats() { return sentenceLevelStats; }


	private void printRunParams(){
		Set<String> keys = runParams.keySet();
		String message = "\n---- Run Data ----\n";

		for (String key: keys){
			message += "# " + key + " " + runParams.get(key) + "\n";
		}
		System.gc();
		logger.info(message);
	}

	private void printFeatureMap(){
		Set<String> keys = featureMap.keySet();
		String message = "\n";

		for (String key: keys){
			message += "# " + key + " " + featureMap.get(key) + "\n";
		}
		System.gc();
		logger.info(message);
	}

	private void validateDirs(){
		String logs = "src/main/java/examples/logs/";
		String output_files = "src/main/java/examples/output_files/";
		String output_files_single_runs = "src/main/java/examples/output_files/single_runs/";

		ArrayList<String> dirs = new ArrayList<>();
		dirs.add(logs);
		dirs.add(output_files);
		dirs.add(output_files_single_runs);

		for (String dir: dirs) {
			File directory = new File(dir);
			if (!directory.exists()) {
				directory.mkdir();
			}
		}
	}

	private long calcExperimentDurationFromRunData(HashMap<String, String> runData) {
		long secondsDiff = 0;
		try {
			String beganExperimentDatetime = runData.get("began experiment datetime");
			String finishExperimentDatetime = runData.get("finish experiment datetime");

			SimpleDateFormat beganExperimentDateFormat = new SimpleDateFormat("HH:mm:ss");
			Date beganExperimentDate = beganExperimentDateFormat.parse(beganExperimentDatetime);

			SimpleDateFormat finishExperimentDateFormat = new SimpleDateFormat("HH:mm:ss");
			Date finishExperimentDate = finishExperimentDateFormat.parse(finishExperimentDatetime);

			secondsDiff = (finishExperimentDate.getTime() - beganExperimentDate.getTime()) / 1000;

		} catch (ParseException e) {
			logger.severe(e.getMessage());
		}
		return secondsDiff;
	}
}


