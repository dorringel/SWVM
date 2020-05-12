package examples;

import com.google.gson.Gson;
import weka.core.stopwords.Null;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class CrossValidation {

    private double EPSILON = 1e-6;

    private static Logger logger = Logger.getLogger(CrossValidation.class.getName());
    private static FileHandler fh = null;

    private HashMap<String, HashMap<String, String>> foldsPaths = new HashMap<String, HashMap<String, String>>();

    // cross validation parameters
    HashMap<String, String> cvParams = new HashMap<>();

    // parameters shared across folds
    HashMap<String, String> runParams = new HashMap<>();

    // data from the various folds (received by running ge.runExperiment())
    private HashMap<String, HashMap<String, String>> foldsRunData = new HashMap<>();
    private HashMap<String, HashMap<String, Map<String, String>>> foldsTrainStats = new HashMap<>();
    private HashMap<String, HashMap<String, Map<String, String>>> foldsTestStats = new HashMap<>();
    private HashMap<String, HashMap<String, Map<String, String>>> foldsDevelopStats = new HashMap<>();

    // sentence level statistics
    private HashMap<String, HashMap<String, Map<String, ArrayList<Double>>>> foldsSentenceLevelStats = new HashMap<>();

    // average statistics
    private HashMap<String, Map<String, String>> avgTrainStats = new HashMap<>();
    private HashMap<String, Map<String, String>> avgTestStats = new HashMap<>();
    private HashMap<String, Map<String, String>> avgDevelopStats = new HashMap<>();

    // fold level statistics
        // key > fold > value
    private HashMap<String, Map<String, String>> foldLevelTrainStats = new HashMap<>();
    private HashMap<String, Map<String, String>> foldLevelTestStats = new HashMap<>();
    private HashMap<String, Map<String, String>> foldLevelDevelopStats = new HashMap<>();

    // entity level statistics
        // key > entity > value
    private HashMap<String, Map<String, String>> entityLevelTrainStats = new HashMap<>();
    private HashMap<String, Map<String, String>> entityLevelTestStats = new HashMap<>();
    private HashMap<String, Map<String, String>> entityLevelDevelopStats = new HashMap<>();

    // fully detailed statistics
    private HashMap<String, Map<String, Map<String, String>>> detailedTrainStats = new HashMap<>();
    private HashMap<String, Map<String, Map<String, String>>> detailedTestStats = new HashMap<>();
    private HashMap<String, Map<String, Map<String, String>>> detailedDevelopStats = new HashMap<>();

    // inverted maps for easier access
    private HashMap<String, List<String>> invertedTrainStats = new HashMap<>();
    private HashMap<String, List<String>> invertedTestStats = new HashMap<>();
    private HashMap<String, List<String>> invertedDevelopStats = new HashMap<>();

    // folds dir
    private String foldsDirPath;

    // cross validation begin datetime
    private static String cvBeginDateTime;

    // output files directories and paths
    private static String outputFilesDir = "src/main/java/examples/output_files/";
    private static String outputFilesCvDir = outputFilesDir + "cv/";
    private static String outputFilesCvDatasetDir;
    private static String outputFilesCvDatasetDatetimeDir;
    private static String outputFilesCvDatasetDatetimePath;

    // logs
    private static String logFilesDir;
    private static String logFilesPath;

    private boolean alignFoldsByFileNumber;
    private String experimentName;
    private int numTrainIterations;
    private String updatorClass;
    private String shortUpdadorClass;
    private String updatorType;
    private String maxFolds;
    private int completedFoldsCounter = 0;
    private String dataset;
    private String createUnsupported;
    private boolean standardMode = false;

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


    public CrossValidation(HashMap<String, String> cvParams) {

        System.out.println("parsing cross validation parameters...");
        try {

            // -------- DEPRECATED PARAMETERS BEGIN HERE -------- //
            // deprecated parameters (still here in case we someday need them)
            cvParams.put("experimentName", "Experiment");
            cvParams.put("updatorType", "single");

            // determine if you cluster Mixed Assignment
            // TODO: make clusterMasFlag=True work
            cvParams.put("clusterMAsFlag", "false");

            // -------- DEPRECATED PARAMETERS END HERE -------- //

            // cross validation parameters
            this.cvParams = cvParams;

            // dataset
            if (cvParams.get("dataset") != null){
                String dtst = cvParams.get("dataset");
                if (dtst.equals("genia") || dtst.equals("genia_small") || dtst.equals("genia_standard") ||
                        dtst.equals("conll2002") || dtst.equals("conll2002_standard") ||
                        dtst.equals("bc2gm") || dtst.equals("bc2gm_standard") ||
                        dtst.equals("conll2000-chunking") || dtst.equals("conll2000-chunking_small_025") || dtst.equals("conll2000-chunking_standard") ||
                        dtst.equals("conll2007-pos") || dtst.equals("conll2007-pos_small_05") || dtst.equals("conll2007-pos_small_025") ||
                        dtst.equals("conll2007-pos_small_01") || dtst.equals("conll2007-pos_small_001") ||
                        dtst.equals("conll2007-pos_standard") ||
                        dtst.equals("genia-pos") || dtst.equals("genia-pos_small_025") || dtst.equals("genia-pos_small_025_2") || dtst.equals("genia-pos_2") ||
                        dtst.equals("synthetic3") || dtst.equals("synthetic4") || dtst.equals("synthetic5")) {
                    dataset = dtst;
                } else {
                throw new IllegalArgumentException("must genia / genia_small / genia_standard / conll2002 / conll2002_standard / bc2gm / bc2gm_standard / " +
                        "conll2000-chunking / conll2000-chunking_small_025 / conll2000-chunking_standard / conll2007-pos" +
                        "conll2007-pos_small_05 / conll2007-pos_small_025 / conll2007-pos_small_01 / conll2007-pos_small_001 / conll2007-pos_standard / " +
                        "genia-pos / genia-pos_small_025 / genia-pos_small_025_2 / genia-pos_2" +
                        "synthetic3 / synthetic4 / synthetic5 for dataset"); }
            } else { throw new IllegalArgumentException("must specify a value for dataset"); }

            // -------- ABSTRACTED PARAMETERS BEGIN HERE -------- //

            // should the cv align train/test/develop files by alphabetical order
            // if this is set to 'false' then the comparisment between different experiments is less valid
            cvParams.put("alignFoldsByFileNumber", "true");

            // createUnsupported
            // -- whether to create all possible features or only those seen in the training set -- /
            /*
                choosing 'true' should provide better generalization, but takes much more time and memory
                rule of thumb: choose 'false' for all data sets where the tag space is greater than 10
                true: genia, conll2002 (3 folds), bc2gm
                false: conll2000-chunking, conll2007-pos
            */
            cvParams.put("createUnsupported", "false");

            if (dataset.equals("genia") || dataset.equals("genia_small") || dataset.equals("genia_standard") ||
                    dataset.equals("conll2002") || dataset.equals("conll2002_standard") ||
                    dataset.equals("bc2gm") || dataset.equals("bc2gm_standard")) {
                createUnsupported = "true";
            } else {
                createUnsupported = "false";
            }

            // this is the version for non-abstracted parameter (when it's in main function)
            /*
            if (cvParams.get("createUnsupported") != null){
                String createUns = cvParams.get("createUnsupported");
                if (createUns.equals("true")){
                    createUnsupported = "true";
                } else if (createUns.equals("false")){
                    createUnsupported = "false";
                } else { throw new IllegalArgumentException("must specify true or false for createUnsupported (string)"); }
            } else{ throw new IllegalArgumentException("must specify a value for createUnsupported (string)"); }
            */

            // standard mode
            if (dataset.equals("genia_standard") ||
                    dataset.equals("bc2gm_standard") ||
                    dataset.equals("conll2002_standard") ||
                    dataset.equals("conll2000-chunking_standard") ||
                    dataset.equals("conll2007-pos_standard")) {
                standardMode = true;
            } else {
                standardMode = false;
            }

            // foldsDirPath

            /// genia
            if (dataset.equals("genia")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia/cv/171225-16-23-54");
            } else if (dataset.equals("genia_small")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia/cv/2fold_small");
            } else if (dataset.equals("genia_standard")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia/cv/standard");
            }

            /// conll2002
            else if (dataset.equals("conll2002")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2002/cv/180102-21-21-42");
            } else if (dataset.equals("conll2002_standard")){
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2002/cv/180115-14-37-50_prepare_for_standard_mode");
            }

            // bc2gm
            else if (dataset.equals("bc2gm")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/bc2gm/cv/180103-17-32-23");
            } else if (dataset.equals("bc2gm_standard")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/bc2gm/cv/180115-14-21-19_prepare_for_standard_mode");
            }

            // conll2000-chunking
            else if (dataset.equals("conll2000-chunking")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2000-chunking/cv/180109-18-38-23");
            } else if (dataset.equals("conll2000-chunking_small_025")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2000-chunking/cv/180109-19-07-05_small_025");
            } else if (dataset.equals("conll2000-chunking_standard")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2000-chunking/cv/180115-14-48-15_prepare_for_standard_mode");
            }

            // conll2007-pos
            else if (dataset.equals("conll2007-pos")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2007-pos/cv/180110-11-36-29");
            } else if (dataset.equals("conll2007-pos_small_05")){
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2007-pos/cv/180110-11-38-26_small_05");
            } else if (dataset.equals("conll2007-pos_small_025")){
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2007-pos/cv/180110-11-37-13_small_025");
            } else if (dataset.equals("conll2007-pos_small_01")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2007-pos/cv/180110-11-35-59_small_01");
            } else if (dataset.equals("conll2007-pos_small_001")){
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2007-pos/cv/180110-12-21-45_small_001");
            } else if (dataset.equals("conll2007-pos_standard")){
                cvParams.put("foldsDirPath", "src/main/resources/data/conll2007-pos/cv/180115-15-03-07_prepare_for_standard_mode");
            }

            // genia-pos
            else if (dataset.equals("genia-pos")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia-pos/cv/180122-15-42-15");
            } else if (dataset.equals("genia-pos_small_025")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia-pos/cv/180122-15-42-12_small_025");
            } else if (dataset.equals("genia-pos_2")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia-pos/cv/180122-11-01-47");
            } else if (dataset.equals("genia-pos_small_025_2")) {
                cvParams.put("foldsDirPath", "src/main/resources/data/genia-pos/cv/180122-11-01-43_small_025");
            }

            // synthetic
            else if (dataset.equals("synthetic3")){
                cvParams.put("foldsDirPath", "src/main/resources/data/synthetic/cv/10000_3_5-20_5");
            } else if (dataset.equals("synthetic4")){
                cvParams.put("foldsDirPath", "src/main/resources/data/synthetic/cv/10000_4_5-20_5");
            } else if (dataset.equals("synthetic5")){
                cvParams.put("foldsDirPath", "src/main/resources/data/synthetic/cv/10000_5_5-20_5");
            }

            // -------- ABSTRACTED PARAMETERS END HERE -------- //

            // folds
            if (cvParams.get("foldsDirPath") != null) {
                foldsDirPath = cvParams.get("foldsDirPath");

                if (foldsDirPath.contains("genia")){
                    // nothing to do
                } else if (foldsDirPath.contains("synthetic")){
                    if (foldsDirPath.contains("10000_3")) {
                        cvParams.put("syntheticDataType", "3");
                        this.cvParams.put("syntheticDataType", "3");
                    }
                    else if (foldsDirPath.contains("10000_4")) {
                        cvParams.put("syntheticDataType", "4");
                        this.cvParams.put("syntheticDataType", "4");
                    }
                    else if (foldsDirPath.contains("10000_5")) {
                        cvParams.put("syntheticDataType", "5");
                        this.cvParams.put("syntheticDataType", "5");
                    }
                }

            } else { throw new IllegalArgumentException("must specify folds file path"); }

            // experiment
            if (cvParams.get("experimentName") != null){
                String expName = cvParams.get("experimentName");
                if (!expName.equals("Experiment")){
                    throw new IllegalArgumentException("currently supporting only Experiment");
                }
                else{
                    experimentName = expName;
                }
            } else{ throw new IllegalArgumentException("must specify experiment name"); }

            // number of training iterations
            if (cvParams.get("numTrainIterations") != null)
            {
                int numIters = Integer.parseInt(cvParams.get("numTrainIterations"));
                if(numIters > 0) {
                    numTrainIterations = numIters;
                } else{ throw new IllegalArgumentException("number of iterations must be a positive integer"); }
            } else {
                logger.log(Level.WARNING, "number of iterations not specified - using default value of 15 iterations");
                numTrainIterations = 15;
            }

            // updator type
            if (cvParams.get("updatorType") != null) {
                String upType = cvParams.get("updatorType");
                if (!upType.equals("single")) {
                    throw new IllegalArgumentException("currently supporting only single mode");
                } else {
                    updatorType = upType;
                }
            } else{ throw new IllegalArgumentException("must specify updator type"); }

            // maximal number of folds to run
            if (cvParams.get("maxFolds") != null) {
                String maxFlds = cvParams.get("maxFolds");
                if (Integer.valueOf(maxFlds) < 0) {
                    throw new IllegalArgumentException("maxFolds has to be a positive integer");
                } else {
                    maxFolds = maxFlds;
                }

            } else{ throw new IllegalArgumentException("must specify maxFolds value"); }

            // updator class
            if (cvParams.get("updatorClass") != null) {
                String updatorCls = cvParams.get("updatorClass");
                if (updatorCls.equals("PerceptronUpdator")){
                    updatorClass = "PerceptronUpdator";
                    shortUpdadorClass = "perceptron";
                }
                else if (updatorCls.equals("KBestMiraUpdator")){
                    updatorClass = "KBestMiraUpdator";
                    shortUpdadorClass = "mira";

                    if (cvParams.get("kMira") != null && Integer.parseInt(cvParams.get("kMira")) > 0){
                        kMira = Integer.parseInt(cvParams.get("kMira"));
                    }
                    else{ throw new IllegalArgumentException("must specify positive Integer K value for K-mira updator"); }
                }

                // SWVPUpdator BEGIN
                else if (updatorCls.equals("SWVPUpdator")) {
                    updatorClass = "SWVPUpdator";
                    shortUpdadorClass = "swvp";

                    // -------- DEPRECATED PARAMETERS BEGIN HERE -------- //
                    if (cvParams.get("clusterMAsFlag") != null) {
                        String clsMAflag = cvParams.get("clusterMAsFlag");

                        if (clsMAflag.equals("true") || (clsMAflag.equals("false"))){
                            clusterMAsFlag = Boolean.parseBoolean(clsMAflag);
                        } else{ throw new IllegalArgumentException("must specify a legal boolean value for clusterMAsFlag (true/false) for JJ updator"); }
                    } else { throw new IllegalArgumentException("clusterMAsFlag value not specified (true/false) for SWVP updator"); }
                    // -------- DEPRECATED PARAMETERS END HERE -------- //

                    // K best inference outputs to train in regards to
                    if (cvParams.get("topK") != null) {
                        int tpK = Integer.valueOf(cvParams.get("topK"));
                        if (tpK >= 1 && tpK <= 20) {
                            topK = tpK;
                        } else {
                            throw new IllegalArgumentException("must specify a value in the range [1,20] for topK"); }
                    } else { throw new IllegalArgumentException("must specify a value top topK"); }

                    // Mixed Assignment type
                    if (cvParams.get("maType") != null && (cvParams.get("maType").equals("passive")) ||
                            (cvParams.get("maType").equals("aggressive")) || (cvParams.get("maType").equals("all"))) {
                        maType = cvParams.get("maType");
                    } else { throw new IllegalArgumentException("must specify legal MA type (passive/aggressive/all) for SWVP updator"); }

                    // JJ substructure type
                    if (cvParams.get("jjType") != null) {
                        String jjT = cvParams.get("jjType");
                        if (jjT.equals("single") || jjT.equals("double") || jjT.equals("triple")){
                            this.jjType = jjT;
                        } else { throw new IllegalArgumentException("must specify 'single' or 'double' or 'triple' for jjType"); }
                    } else { throw new IllegalArgumentException("must specify a value for jjType"); }

                    // gamma calculation - method of calculation (uniform, wm, wmr, softmax...)
                    if (cvParams.get("gammaCalculationMethod") != null) {
                        String gammaClcMethod = cvParams.get("gammaCalculationMethod");
                        if (gammaClcMethod.equals("uniform") ||
                                gammaClcMethod.equals("wm")      || gammaClcMethod.equals("wmr") ||
                                gammaClcMethod.equals("softmax") || gammaClcMethod.equals("softmin") ||
                                gammaClcMethod.equals("one") || gammaClcMethod.equals("opt")) {
                            gammaCalculationMethod = gammaClcMethod;
                        } else { throw new IllegalArgumentException("gammaCalculationMethod can only be uniform or wm or wmr " +
                                "or softmax or softmin or one or opt"); }
                    } else { throw new IllegalArgumentException("must specify gammaCalculationMethod"); }

                    // gamma calculation - value of Beta hyper-parameter
                    if (cvParams.get("gammaCalculationBeta") != null) {
                        double gammaClcBeta = Double.valueOf(cvParams.get("gammaCalculationBeta"));
                        if (gammaClcBeta > 0.1 - EPSILON && gammaClcBeta < 10 + EPSILON) {
                            gammaCalculationBeta = gammaClcBeta;
                        } else { throw new IllegalArgumentException("gammaCalculationBeta must be in the range [0.1,10]"); }
                    } else { throw new IllegalArgumentException("must specify gammaCalculationBeta"); }

                    // gamma calculation - objective function type (min/max)
                    if (cvParams.get("gammaCalculationObjectiveType") != null){
                        String objType = cvParams.get("gammaCalculationObjectiveType");
                        if (objType.equals("MINIMIZE") || objType.equals("MAXIMIZE")){
                            gammaCalculationObjectiveType = objType;
                        } else { throw new IllegalArgumentException("must specify MINIMIZE or MAXIMIZE for gammaCalculationObjectiveType"); }
                    } else{ throw new IllegalArgumentException("must specify legal objective type (MAXIMIZE/MINIMIZE) for SWVP updator"); }

                }
                // SWVPUpdator END

                // SWVMUpdator BEGIN
                else if (updatorCls.equals("SWVMUpdator")) {
                    updatorClass = "SWVMUpdator";
                    shortUpdadorClass = "swvm";

                    // -------- DEPRECATED PARAMETERS BEGIN HERE -------- //
                    if (cvParams.get("clusterMAsFlag") != null) {
                        String clsMAflag = cvParams.get("clusterMAsFlag");

                        if (clsMAflag.equals("true") || (clsMAflag.equals("false"))){
                            clusterMAsFlag = Boolean.parseBoolean(clsMAflag);
                        } else{ throw new IllegalArgumentException("must specify a legal boolean value for clusterMAsFlag (true/false) for JJ updator"); }
                    } else { throw new IllegalArgumentException("clusterMAsFlag value not specified (true/false) for SWVM updator"); }
                    // -------- DEPRECATED PARAMETERS END HERE -------- //

                    // K best inference outputs to train in regards to
                    if (cvParams.get("topK") != null) {
                        int tpK = Integer.valueOf(cvParams.get("topK"));
                        if (tpK >= 1 && tpK <= 20) {
                            topK = tpK;
                        } else {
                            throw new IllegalArgumentException("must specify a value in the range [1,20] for topK"); }
                    } else { throw new IllegalArgumentException("must specify a value top topK"); }

                    // Mixed Assignment type
                    if (cvParams.get("maType") != null && (cvParams.get("maType").equals("passive")) ||
                            (cvParams.get("maType").equals("aggressive")) || (cvParams.get("maType").equals("all"))) {
                        maType = cvParams.get("maType");
                    }
                    else { throw new IllegalArgumentException("must specify legal MA type (passive/aggressive/all) for SWVM updator"); }

                    // JJ substructure type
                    if (cvParams.get("jjType") != null) {
                        String jjT = cvParams.get("jjType");
                        if (jjT.equals("single") || jjT.equals("double") || jjT.equals("triple")){
                            this.jjType = jjT;
                        } else { throw new IllegalArgumentException("must specify 'single' or 'double' or 'triple' for jjType"); }
                    } else { throw new IllegalArgumentException("must specify a value for jjType"); }

                    // gamma calculation - method of calculation (uniform, wm, wmr, softmax...)
                    if (cvParams.get("gammaCalculationMethod") != null) {
                        String gammaClcMethod = cvParams.get("gammaCalculationMethod");
                        if (gammaClcMethod.equals("uniform") ||
                                gammaClcMethod.equals("wm")      || gammaClcMethod.equals("wmr") ||
                                gammaClcMethod.equals("softmax") || gammaClcMethod.equals("softmin") ||
                                gammaClcMethod.equals("one") || gammaClcMethod.equals("opt")) {
                            gammaCalculationMethod = gammaClcMethod;
                        } else { throw new IllegalArgumentException("gammaCalculationMethod can only be uniform or wm or wmr " +
                                    "or softmax or softmin or one or opt"); }
                    } else { throw new IllegalArgumentException("must specify gammaCalculationMethod"); }

                    // gamma calculation - value of Beta hyper-parameter
                    if (cvParams.get("gammaCalculationBeta") != null) {
                        double gammaClcBeta = Double.valueOf(cvParams.get("gammaCalculationBeta"));
                        if (gammaClcBeta > 0.1 - EPSILON && gammaClcBeta < 10 + EPSILON) {
                            gammaCalculationBeta = gammaClcBeta;
                        } else { throw new IllegalArgumentException("gammaCalculationBeta must be in the range [0.1,10]"); }
                    } else { throw new IllegalArgumentException("must specify gammaCalculationBeta"); }

                    // gamma calculation - objective function type (min/max)
                    if (cvParams.get("gammaCalculationObjectiveType") != null){
                        String objType = cvParams.get("gammaCalculationObjectiveType");
                        if (objType.equals("MINIMIZE") || objType.equals("MAXIMIZE")){
                            gammaCalculationObjectiveType = objType;
                        } else { throw new IllegalArgumentException("must specify MINIMIZE or MAXIMIZE for gammaCalculationObjectiveType"); }
                    } else{ throw new IllegalArgumentException("must specify legal objective type (MAXIMIZE/MINIMIZE) for SWVM updator"); }
                }
                // SWVMUpdator END


                else{ throw new IllegalArgumentException("illegal updator class"); }
            } else { throw new IllegalArgumentException("must specify an updator class"); }

            // feature map
            HashMap<String, Boolean> tempFeatureMap = _getFeatureMap();
            if (tempFeatureMap != null)
            {
                this.featureMap = tempFeatureMap;
            } else{ throw new IllegalArgumentException("must provide a non-null feature map"); }

            // cross validation begin time
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyMMdd-HH-mm-ss");
            cvBeginDateTime = dateFormat.format(new Date());

            if (cvParams.get("alignFoldsByFileNumber") != null) {
                String alignFlsByNmbr = cvParams.get("alignFoldsByFileNumber");
                if (alignFlsByNmbr.equals("true")){
                    alignFoldsByFileNumber = true;
                } else if (alignFlsByNmbr.equals("false")){
                    alignFoldsByFileNumber = false;
                } else{ throw new IllegalArgumentException("align files by number must be 'true' or 'false"); }
            } else { throw new IllegalArgumentException("must specify a alignFoldsByFileNumber"); }

        } catch (Exception e) {
            logger.log(Level.SEVERE, e.getMessage());
            System.exit(1);
        }

        // validate the existence of required directories
        this._validateDirs();

        // initialize logger
        this._initLogger();
    }

    private void _initLogger(){
        try {
            fh=new FileHandler(this.logFilesPath, true);
        } catch (SecurityException | IOException e) {
            e.printStackTrace();
        }
        SimpleFormatter sf = new SimpleFormatter();
        fh.setFormatter(sf);
        fh.setLevel(Level.ALL);
        logger.addHandler(fh);
        logger.setLevel(Level.ALL);
//        Logger l = Logger.getLogger("");
//        SimpleFormatter sf = new SimpleFormatter();
//        //sf.format="%4$: %5$ [%1$tc]%n";  // TODO: create a log properties file with a single lines format
//        fh.setFormatter(sf);
//        l.addHandler(fh);
//        l.setLevel(Level.ALL);
    }

    private void _validateDirs() {

        String datasetType = dataset + '/';

        outputFilesCvDatasetDir = outputFilesCvDir + datasetType;
        outputFilesCvDatasetDatetimeDir = outputFilesCvDatasetDir + cvBeginDateTime + "_" + maxFolds + "_" + shortUpdadorClass + "/";
        outputFilesCvDatasetDatetimePath = outputFilesCvDatasetDatetimeDir + "__run";

        logFilesDir = outputFilesCvDatasetDatetimeDir + "logs/";
        logFilesPath = logFilesDir + "log_file.log";


        ArrayList<String> dirs = new ArrayList<>();
        dirs.add(outputFilesDir);
        dirs.add(outputFilesCvDir);
        dirs.add(outputFilesCvDatasetDir);
        dirs.add(outputFilesCvDatasetDatetimeDir);
        dirs.add(logFilesDir);

        for (String dir: dirs) {
            File directory = new File(dir);
            if (!directory.exists()) {
                directory.mkdir();
            }
        }
    }

    private void validateFiles(){
        logger.info("validating fold files...");
        try {
            File dir = new File(foldsDirPath);
            File[] directoryListing0 = dir.listFiles();

            // handle ds_store file in cv folds directory
            ArrayList<File> removeList = new ArrayList<>();
            ArrayList<File> dirArrayList = new ArrayList<>(Arrays.asList(directoryListing0));
            for (File file : dirArrayList){
                String fileStr = file.toString().toLowerCase();
                if (fileStr.contains("ds_store")){
                    removeList.add(file);
                } else if (fileStr.contains("duplication_cleanup_report")){
                    removeList.add(file);
                }
            }

            for (File file : removeList) {
                dirArrayList.remove(file);
            }


            int dirSize = dirArrayList.size();
            File[] directoryListing = new File[dirSize];
            directoryListing = dirArrayList.toArray(directoryListing);


            // align train/test/develop by their file system alphabetical order
            if (alignFoldsByFileNumber) {
                Arrays.sort(directoryListing);
            }

            if (directoryListing.length % 3 !=0){ throw new IllegalArgumentException("number of files must be a multiple of thee (train,test,develop)"); }
            int numFolds = (int)directoryListing.length/3;
            logger.info("there are " + String.valueOf(numFolds) + " folds in the specified directory");


            for (int fld = 1; fld <= numFolds; fld++){
                HashMap<String, String> emptyPathMap = new HashMap<>();
                foldsPaths.put(String.valueOf(fld), emptyPathMap);
            }

            int trainFileCounter = 0;
            int testFileCounter = 0;
            int developFileCounter = 0;

            if (directoryListing != null) {

                for (File fileName : directoryListing) {
                    String fileNameStr = fileName.toString();
                    if (fileNameStr.contains("_train")) {
                        trainFileCounter++;
                        HashMap<String, String> currPathsMap = foldsPaths.get(String.valueOf(trainFileCounter));
                        currPathsMap.put("train", fileNameStr);
                        foldsPaths.put(String.valueOf(trainFileCounter), currPathsMap);
                    }
                    if (fileNameStr.contains("_test")) {
                        testFileCounter++;
                        HashMap<String, String> currPathsMap = foldsPaths.get(String.valueOf(testFileCounter));
                        currPathsMap.put("test", fileNameStr);
                        foldsPaths.put(String.valueOf(testFileCounter), currPathsMap);
                    }
                    if (fileNameStr.contains("_develop")) {
                        developFileCounter++;
                        HashMap<String, String> currPathsMap = foldsPaths.get(String.valueOf(developFileCounter));
                        currPathsMap.put("develop", fileNameStr);
                        foldsPaths.put(String.valueOf(developFileCounter), currPathsMap);
                    }
                }
            if (trainFileCounter != testFileCounter ||
                trainFileCounter != developFileCounter ||
                testFileCounter != developFileCounter){
                    throw new IllegalArgumentException("number of fold files for each data group (train, test, develop) must be the same"); }

            // Add number of folds to cvParams
            this.cvParams.put("numFolds", String.valueOf(trainFileCounter));

            } else{
                throw new IllegalArgumentException("must specify a non empty directory");
                // TODO: perhaps a deeper inspection (names are not aligned or something)
            }
        } catch (Exception e) {
            logger.log(Level.SEVERE, e.getMessage());
            logger.severe("Fold file structure is missing or not as expected");
            logger.severe("look at the dataset you chose and make sure the files are actually in the directory");
            System.exit(1);
        }
    }

    private void printCvParams(){
        logger.info("printing Cross-Validation parameters...");

        Set<String> keys = cvParams.keySet();
        String message = "\n---- Cross-Validation Parameters ----\n";

        for (String key: keys){
            message += "# " + key + " " + cvParams.get(key) + "\n";
        }
        System.gc();
        logger.info(message);
    }

    private void performCrossValidation(){
        logger.info("performing Cross-Validation...");

        runParams.put("logFilesPath", logFilesPath);
        runParams.put("resultsFilePath", outputFilesCvDatasetDatetimePath);
        runParams.put("numTrainIterations", String.valueOf(numTrainIterations));
        runParams.put("createUnsupported", String.valueOf(createUnsupported));

        if (updatorClass.equals("PerceptronUpdator")){
            runParams.put("updatorClass", "PerceptronUpdator");
        }
        else if (updatorClass.equals("KBestMiraUpdator")){
            runParams.put("updatorClass", "KBestMiraUpdator");
            runParams.put("kMira", String.valueOf(kMira));
        }
        else if (updatorClass.equals("SWVPUpdator")){
            runParams.put("updatorClass", "SWVPUpdator");
            // deprecated begin
            runParams.put("clusterMAsFlag", Boolean.toString(clusterMAsFlag));
            // deprecated end
            runParams.put("topK", String.valueOf(topK));
            runParams.put("maType", maType);
            runParams.put("jjType", String.valueOf(jjType));
            runParams.put("gammaCalculationMethod", gammaCalculationMethod);
            runParams.put("gammaCalculationBeta", String.valueOf(gammaCalculationBeta));
            runParams.put("gammaCalculationObjectiveType", gammaCalculationObjectiveType);
        }
        else if (updatorClass.equals("SWVMUpdator")){
            runParams.put("updatorClass", "SWVMUpdator");
            // deprecated begin
            runParams.put("clusterMAsFlag", Boolean.toString(clusterMAsFlag));
            // deprecated end
            runParams.put("topK", String.valueOf(topK));
            runParams.put("maType", maType);
            runParams.put("jjType", String.valueOf(jjType));
            runParams.put("gammaCalculationMethod", gammaCalculationMethod);
            runParams.put("gammaCalculationBeta", String.valueOf(gammaCalculationBeta));
            runParams.put("gammaCalculationObjectiveType", gammaCalculationObjectiveType);
        }

        int actualNumFolds = Math.min(foldsPaths.size(), Integer.valueOf(maxFolds));
        cvParams.put("actualNumFolds", String.valueOf(actualNumFolds));

        for(int fld = 1; fld <= actualNumFolds; fld++){
            logger.info("began Cross-Validation Fold #" + String.valueOf(fld));
            // specify train/test files according to current fold
            String currTrainFilePath = foldsPaths.get(String.valueOf(fld)).get("train");
            String currTestFilePath = foldsPaths.get(String.valueOf(fld)).get("test");
            String currDevelopFilePath = foldsPaths.get(String.valueOf(fld)).get("develop");
            runParams.put("trainFilePath", currTrainFilePath);
            runParams.put("testFilePath", currTestFilePath);
            runParams.put("developFilePath", currDevelopFilePath);

            runParams.put("completedFoldsCounter", String.valueOf(completedFoldsCounter));

            String exceptionCauseStr;
            try {
                Experiment ge = new Experiment(runParams, featureMap, logger);
                ge.runExperiment();

                // get run data, train statistics, test statistics and sentence-level statistics for the current fold
                String fldStr = String.valueOf(fld);
                foldsRunData.put(fldStr, ge.getRunData());
                foldsTrainStats.put(fldStr, ge.getTrainStats());
                foldsTestStats.put(fldStr, ge.getTestStats());
                foldsDevelopStats.put(fldStr, ge.getDevelopStats());
                foldsSentenceLevelStats.put(fldStr, ge.getSentenceLevelStats());

                // increment completed folds counter
                completedFoldsCounter++;

                // process results of folds completed so far
                processResults();

                // write average results to file
                writeAvgResultsToFile();

                // write detailed results to file
                writeDetailedResultsToFile();

                // write sentence level results to file
                writeSentenceLevelStatsToFile(fldStr);

                // write duration time of the experiment (the fold) which has just been completed to file
                writeExperimentDurationToFile(fldStr, foldsRunData.get(fldStr));
            } catch (Exception e){
                exceptionCauseStr = e.getCause().toString();
                logger.log(Level.SEVERE, exceptionCauseStr);
                System.exit(1);
          //  } finally {
                //writeCrashDuringCVToFile(exceptionCauseStr);
            }
        }
    }

    /**
     * writes sentence level results to file.
     * @param fldStr : the number of iteration to write
     *
     *               will typically be called after each fold is completed
     *               and will write current fold's statistics
     */
    private void writeSentenceLevelStatsToFile(String fldStr){
        logger.info("writing sentence-level statistics...");

            String fileName = outputFilesCvDatasetDatetimeDir + _getOutputFileName("sentence") + ".json";


            try{
                PrintWriter writer = new PrintWriter(fileName, "UTF-8");

                Gson gson = new Gson();

                HashMap<String, Map<String, ArrayList<Double>>> fldSentenceLevelStats =
                        foldsSentenceLevelStats.get(fldStr);
                String json = gson.toJson(fldSentenceLevelStats);
                writer.println(json);

                writer.close();
            } catch (IOException e) {
                logger.severe(e.getMessage());
            }
        }

    private void processResults(){
        logger.info("processing results from experiments...");

        // set of metrics (usually Accuracy, Precision, Recall, F1)
        Set keySet = foldsTrainStats.get("1").keySet();

        // number of folds of the CV
        int numFolds = (int)foldsTrainStats.size();

        // fulll
        // set of entities (in Genia its protein, RNA, DNA, etc.)

        Object keyst = foldsTrainStats.get("1").get("F1");
        if (keyst == null) {
            throw new NullPointerException("F1 is not in foldsTrainStats");
        }

        Set entitySet = foldsTrainStats.get("1").get("F1").keySet();

        // key > fold > entity > value
        // "Precision" > "1" > "protein" > 0.86674

        for(Object key : keySet){
            String keyStr = String.valueOf(key);
            double keyTrainAvg = 0;
            double keyTestAvg = 0;
            double keyDevelopAvg = 0;

            List<String> keyTrainArrayList = new ArrayList<>();
            List<String> keyTestArrayList = new ArrayList<>();
            List<String> keyDevelopArrayList = new ArrayList<>();

            Map<String, String> keyTrainHashMap = new HashMap<>();
            Map<String, String> keyTestHashMap = new HashMap<>();
            Map<String, String> keyDevelopHashMap = new HashMap<>();

            // fulll
            Map<String, Map<String, String>> keyFoldTrainHashMap = new HashMap<>();
            Map<String, Map<String, String>> keyFoldTestHashMap = new HashMap<>();
            Map<String, Map<String, String>> keyFoldDevelopHashMap = new HashMap<>();

            for(int fld = 1; fld <= numFolds; fld++){
                String fldStr = String.valueOf(fld);

                Map<String, String> keyFoldEntityTrainHashMap = new HashMap<>();
                Map<String, String> keyFoldEntityTestHashMap = new HashMap<>();
                Map<String, String> keyFoldEntityDevelopHashMap = new HashMap<>();

                // fulll
                for(Object entity : entitySet){
                    if (keyStr.equals("Accuracy") && !entity.equals("overall"))
                        // for Accuracy we only have "overall"
                        continue;
                    // get statistic for specific key, fold and entity type
                    String trValue = foldsTrainStats.get(fldStr).get(keyStr).get(entity);
                    String teValue = foldsTestStats.get(fldStr).get(keyStr).get(entity);
                    String deValue = foldsDevelopStats.get(fldStr).get(keyStr).get(entity);

                    keyFoldEntityTrainHashMap.put((String)entity, trValue);
                    keyFoldEntityTestHashMap.put((String)entity, teValue);
                    keyFoldEntityDevelopHashMap.put((String)entity, deValue);
                }

                // fulll
                keyFoldTrainHashMap.put(fldStr, keyFoldEntityTrainHashMap);
                keyFoldTestHashMap.put(fldStr, keyFoldEntityTestHashMap);
                keyFoldDevelopHashMap.put(fldStr, keyFoldEntityDevelopHashMap);


                // get "overall" statistic for specific key and fold
                String trValue = foldsTrainStats.get(fldStr).get(keyStr).get("overall");
                String teValue = foldsTestStats.get(fldStr).get(keyStr).get("overall");
                String deValue = foldsDevelopStats.get(fldStr).get(keyStr).get("overall");

                // accumulate for average (per CV)
                keyTrainAvg += Double.valueOf(trValue);
                keyTestAvg += Double.valueOf(teValue);
                keyDevelopAvg += Double.valueOf(deValue);

                // append to ArrayList (per fold)
                keyTrainArrayList.add(trValue);
                keyTestArrayList.add(teValue);
                keyDevelopArrayList.add(deValue);

                // put in HashMap (per fold)
                keyTrainHashMap.put(fldStr, trValue);
                keyTestHashMap.put(fldStr, teValue);
                keyDevelopHashMap.put(fldStr, deValue);
            }

            // fulll
            detailedTrainStats.put(keyStr, keyFoldTrainHashMap);
            detailedTestStats.put(keyStr, keyFoldTestHashMap);
            detailedDevelopStats.put(keyStr, keyFoldDevelopHashMap);


            // calc average
            keyTrainAvg = keyTrainAvg/numFolds;
            keyTestAvg = keyTestAvg/numFolds;
            keyDevelopAvg = keyDevelopAvg/numFolds;

            // build "overall" average per key HashMap
            HashMap<String, String> keyTrainAvgOverallMap = new HashMap<>();
            HashMap<String, String> keyTestAvgOverallMap = new HashMap<>();
            HashMap<String, String> keyDevelopAvgOverallMap = new HashMap<>();
            keyTrainAvgOverallMap.put("overall", String.valueOf(keyTrainAvg));
            keyTestAvgOverallMap.put("overall", String.valueOf(keyTestAvg));
            keyDevelopAvgOverallMap.put("overall", String.valueOf(keyDevelopAvg));


            // put average HashMap in final object
            avgTrainStats.put(keyStr, keyTrainAvgOverallMap);
            avgTestStats.put(keyStr, keyTestAvgOverallMap);
            avgDevelopStats.put(keyStr, keyDevelopAvgOverallMap);

            // put detailed ArrayList in final object
            invertedTrainStats.put(keyStr, keyTrainArrayList);
            invertedTestStats.put(keyStr, keyTestArrayList);
            invertedDevelopStats.put(keyStr, keyDevelopArrayList);

            // put detailed ArrayList in final object
            foldLevelTrainStats.put(keyStr, keyTrainHashMap);
            foldLevelTestStats.put(keyStr, keyTestHashMap);
            foldLevelDevelopStats.put(keyStr, keyDevelopHashMap);
        }
    }

    private void processEntityLevelResults() {
        logger.info("processing entity-level results from experiments...");

        // set of metrics (usually Accuracy, Precision, Recall, F1)
        Set keySet = foldsTrainStats.get("1").keySet();

        // number of folds of the CV
        int numFolds = (int) foldsTrainStats.size();

        // fulll
        // set of entities (in Genia its protein, RNA, DNA, etc.)

        Object keyst = foldsTrainStats.get("1").get("F1");
        if (keyst == null) {
            throw new NullPointerException("F1 is not in foldsTrainStats");
        }

        Set entitySet = foldsTrainStats.get("1").get("F1").keySet();

        // key > entity > value
        // "Precision" > "protein" > 0.86674

        for (Object key : keySet) {
            String keyStr = String.valueOf(key);

            Map<String, String> entityTrainStats = new HashMap<>();
            Map<String, String> entityTestStats = new HashMap<>();
            Map<String, String> entityDevelopStats = new HashMap<>();

            for (Object entity : entitySet) {
                if (keyStr.equals("Accuracy") && !entity.equals("overall"))
                    // for Accuracy we only have "overall"
                    continue;

                String entityStr = String.valueOf(entity);

                double foldTrainAverage = 0.0;
                double foldTestAverage = 0.0;
                double foldDevelopAverage = 0.0;

                for (int fld = 1; fld <= numFolds; fld++) {
                    String fldStr = String.valueOf(fld);

                    // get statistic for specific key, fold and entity type
                    String trValue = foldsTrainStats.get(fldStr).get(keyStr).get(entity);
                    String teValue = foldsTestStats.get(fldStr).get(keyStr).get(entity);
                    String deValue = foldsDevelopStats.get(fldStr).get(keyStr).get(entity);

                    // deal with the case when you have an entity which only exists
                    // in one of the files (only on train, for example)
                    if (trValue == null || trValue == "null"){
                        trValue = "0.0";
                    } if (teValue == null || trValue == "null"){
                        teValue = "0.0";
                    } if (deValue == null || trValue == "null"){
                        deValue = "0.0";
                    }

                    foldTrainAverage += Double.valueOf(trValue);
                    foldTestAverage += Double.valueOf(teValue);
                    foldDevelopAverage += Double.valueOf(deValue);
                }
                foldTrainAverage = foldTrainAverage/numFolds;
                foldTestAverage = foldTestAverage/numFolds;
                foldDevelopAverage = foldDevelopAverage/numFolds;

                entityTrainStats.put(entityStr, String.valueOf(foldTrainAverage));
                entityTestStats.put(entityStr, String.valueOf(foldTestAverage));
                entityDevelopStats.put(entityStr, String.valueOf(foldDevelopAverage));
            }

            entityLevelTrainStats.put(keyStr, entityTrainStats);
            entityLevelTestStats.put(keyStr, entityTestStats);
            entityLevelDevelopStats.put(keyStr, entityDevelopStats);
        }
    }

    private void writeAvgResultsToFile(){
        logger.info("writing average results to file...");

        HashMap<String, String> runData = this.runParams;

        String fileName = outputFilesCvDatasetDatetimeDir + _getOutputFileName("avg") + ".txt";


        try{
            PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            writer.println("---- THIS IS A CROSS VALIDATION RUN ----");
            writer.println("---- RUNNING AFTER " + String.valueOf(completedFoldsCounter) + " FOLDS HAVE COMPLETED ----\n");
            writer.println("---- AVERAGE INFORMATION ----\n\n");

            writer.println("---- Begin Datetime ----");
            writer.println(cvBeginDateTime);
            writer.println("\n");

            writer.println("---- Updator class ----");
            writer.println(runData.get("updatorClass"));
            writer.println("\n");

            writer.println("---- Easy Excel Copy-paste ----");
            String easyCopyString = cvBeginDateTime + '\t' +
                    avgTestStats.get("Accuracy").get("overall") + '\t' + avgTestStats.get("F1").get("overall") + '\t' +
                    avgDevelopStats.get("Accuracy").get("overall") + '\t' + avgDevelopStats.get("F1").get("overall") + '\t'  +
                    avgTrainStats.get("Accuracy").get("overall") + '\t'  + avgTrainStats.get("F1").get("overall");

            writer.println(easyCopyString);
            writer.println("\n");

            writer.println("---- Bottom lines ----");
            writer.println("test Accuracy: " + avgTestStats.get("Accuracy").get("overall"));
            writer.println("test F1: " + avgTestStats.get("F1").get("overall"));
            writer.println("\n");
            writer.println("develop Accuracy: " + avgDevelopStats.get("Accuracy").get("overall"));
            writer.println("develop F1: " + avgDevelopStats.get("F1").get("overall"));
            writer.println("\n");
            writer.println("train Accuracy: " + avgTrainStats.get("Accuracy").get("overall"));
            writer.println("train F1: " + avgTrainStats.get("F1").get("overall"));
            writer.println("\n");

            writer.println("---- Run Data ----");
            for (Map.Entry<String, String> entry : runData.entrySet()){
                String key = entry.getKey();
                String value = entry.getValue();
                if (key.equals("clusterMAsFlag")) {
                    continue;
                } else {
                    writer.println(key + ": " + value);
                }
            }
            writer.println("\n");

            writer.println("---- Cross-Validation Parameters ----");
            for (Map.Entry<String, String> entry : cvParams.entrySet()){
                String key = entry.getKey();
                String value = entry.getValue();
                if (key.equals("clusterMAsFlag") || key.equals("updatorType") || key.equals("numFolds") || key.equals("alignFoldsByFileNumber")) {
                    continue;
                } else {
                    writer.println(key + ": " + value);
                }
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

            for (Map.Entry<String, Map<String, String>> entry : avgTrainStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();
                writer.println(metric + ": " + val.get("overall"));
            }

            writer.println("---- Train End ----");
            writer.println("\n");
            writer.println("---- Test Begin ----");
            //writer.println("Accuracy: " + testStats.get("Accuracy").get("overall"));

            for (Map.Entry<String, Map<String, String>> entry : avgTestStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();
                writer.println(metric + ": " + val.get("overall"));
            }

            writer.println("---- Test End ----");
            writer.println("\n");
            writer.println("---- Develop Begin ----");
            //writer.println("Accuracy: " + developStats.get("Accuracy").get("overall"));

            for (Map.Entry<String, Map<String, String>> entry : avgDevelopStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();
                writer.println(metric + ": " + val.get("overall"));
            }

            writer.println("---- Develop End ----");
            writer.close();
        } catch (IOException e) {
            logger.severe(e.getMessage());
        }
    }

    private void writeDetailedResultsToFile(){
        logger.info("writing detailed results to file...");

        HashMap<String, String> runData = this.runParams;
        int numFolds = (int)foldsTrainStats.size();

        // set of entities (in Genia its protein, RNA, DNA, etc.)
        Set entitySet = foldsTrainStats.get("1").get("F1").keySet();


        String fileName = outputFilesCvDatasetDatetimeDir + _getOutputFileName("fld") + ".txt";


        try{
            PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            writer.println("---- THIS IS A CROSS VALIDATION RUN ----");
            writer.println("---- RUNNING AFTER " + String.valueOf(completedFoldsCounter) + " FOLDS HAVE COMPLETED ----\n");
            writer.println("---- FOLD LEVEL INFORMATION ----\n\n");

            writer.println("---- Begin Datetime ----");
            writer.println(cvBeginDateTime);
            writer.println("\n");

            writer.println("---- Updator class ----");
            writer.println(runData.get("updatorClass"));
            writer.println("\n");

            writer.println("---- Easy Excel Copy-paste ----");
            String easyCopyString = cvBeginDateTime + '\t' +
                    avgTestStats.get("Accuracy").get("overall") + '\t' + avgTestStats.get("F1").get("overall") + '\t' +
                    avgDevelopStats.get("Accuracy").get("overall") + '\t' + avgDevelopStats.get("F1").get("overall") + '\t'  +
                    avgTrainStats.get("Accuracy").get("overall") + '\t'  + avgTrainStats.get("F1").get("overall");

            writer.println(easyCopyString);
            writer.println("\n");

            writer.println("---- Bottom lines ----");
            writer.println("test Accuracy: " + avgTestStats.get("Accuracy").get("overall"));
            writer.println("test F1: " + avgTestStats.get("F1").get("overall"));
            writer.println("\n");
            writer.println("develop Accuracy: " + avgDevelopStats.get("Accuracy").get("overall"));
            writer.println("develop F1: " + avgDevelopStats.get("F1").get("overall"));
            writer.println("\n");
            writer.println("train Accuracy: " + avgTrainStats.get("Accuracy").get("overall"));
            writer.println("train F1: " + avgTrainStats.get("F1").get("overall"));
            writer.println("\n");

            writer.println("---- Folds Run-times and File-names Begin ----");

            writer.println("- This module provides the run time for each experiment (each fold of the CV) ");
            writer.println("- In the following format:");
            writer.println("-- " + "fold_number" + ": " + "time-in-minutes" + " (" + "time-in-seconds" + ")");
            writer.println("-- trainFilePath");
            writer.println("-- testFilePath");
            writer.println("-- developFilePath");

            for (HashMap.Entry<String, HashMap<String, String>> entry : foldsRunData.entrySet()) {
                String fldStr = entry.getKey();
                HashMap<String, String> fldRunData = entry.getValue();

                long secondsDiff = _calcExperimentDurationFromRunData(fldRunData);
                long minutesDiff = secondsDiff / 60;

                String trainFilePath = fldRunData.get("train file pathname");
                String testFilePath = fldRunData.get("test file pathname");
                String developFilePath = fldRunData.get("develop file pathname");

                writer.println(fldStr + ": " + String.valueOf(minutesDiff) + "m (" + String.valueOf(secondsDiff) + "s)");
                writer.println(trainFilePath);
                writer.println(testFilePath);
                writer.println(developFilePath);
                writer.println("\n");
            }
            writer.println("---- Folds Run-times and File-names End ----");
            writer.println("\n");

            writer.println("---- Train Entity-level Statistics Begin ----");

            writer.println("- This module provides statistics for each entity");
            writer.println("- averaged over all folds of the cross-validation\n");
            for (HashMap.Entry<String, Map<String, String>> entry : entityLevelTrainStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();

                writer.println("# " + metric);
                for(Object entity : entitySet) {
                    String entityStr = String.valueOf(entity);

                    if (metric.equals("Accuracy") && !entityStr.equals("overall"))
                        // for Accuracy we only have "overall"
                        continue;

                    String currVal = val.get(entityStr);
                    writer.println(entityStr + ": " + currVal);
                }
                writer.print("\n");
            }

            writer.println("---- Train Entity-level Statistics End ----");
            writer.println("\n");

            writer.println("---- Test Entity-level Statistics Begin ----");

            writer.println("- This module provides statistics for each entity");
            writer.println("- averaged over all folds of the cross-validation\n");

            for (HashMap.Entry<String, Map<String, String>> entry : entityLevelTestStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();

                writer.println("# " + metric);
                for(Object entity : entitySet) {
                    String entityStr = String.valueOf(entity);

                    if (metric.equals("Accuracy") && !entityStr.equals("overall"))
                        // for Accuracy we only have "overall"
                        continue;

                    String currVal = val.get(entityStr);
                    writer.println(entityStr + ": " + currVal);
                }
                writer.print("\n");
            }

            writer.println("---- Test Entity-level Statistics End ----");
            writer.println("\n");

            writer.println("---- Develop Entity-level Statistics Begin ----");

            writer.println("- This module provides statistics for each entity");
            writer.println("- averaged over all folds of the cross-validation\n");

            for (HashMap.Entry<String, Map<String, String>> entry : entityLevelDevelopStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();

                writer.println("# " + metric);
                for(Object entity : entitySet) {
                    String entityStr = String.valueOf(entity);

                    if (metric.equals("Accuracy") && !entityStr.equals("overall"))
                        // for Accuracy we only have "overall"
                        continue;

                    String currVal = val.get(entityStr);
                    writer.println(entityStr + ": " + currVal);
                }
                writer.print("\n");
            }

            writer.println("---- Develop Entity-level Statistics End ----");
            writer.println("\n");

            writer.println("---- Train Fold-level Statistics Begin ----");

            writer.println("- This module provides statistics for each fold");
            writer.println("- averaged over all entity types\n");

            for (HashMap.Entry<String, Map<String, String>> entry : foldLevelTrainStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();

                writer.println("# " + metric);
                String keyOverallAvg = avgTrainStats.get(metric).get("overall");
                String keyOverallStd = _calcStdStringFromStringList(invertedTrainStats.get(metric));
                String keyOverallMed = _calcMedianStringFromStringList(invertedTrainStats.get(metric));

                writer.println("s: " + keyOverallStd + " (std)");
                writer.println("m: " + keyOverallMed + " (median)\n");
                writer.println("a: " + keyOverallAvg + " (average)");

                for(int fld = 1; fld <= numFolds; fld++) {
                    String fldStr = String.valueOf(fld);
                    String currVal = val.get(fldStr);
                    writer.println(fldStr + ": " + currVal);
                }
                writer.print("\n");
            }
            writer.println("---- Train Fold-level Statistics End ----");
            writer.println("\n");

            writer.println("---- Test Fold-level Statistics Begin ----");

            writer.println("- This module provides statistics for each fold");
            writer.println("- averaged over all entity types\n");

            for (HashMap.Entry<String, Map<String, String>> entry : foldLevelTestStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();

                writer.println("# " + metric);
                String keyOverallAvg = avgTestStats.get(metric).get("overall");
                String keyOverallStd = _calcStdStringFromStringList(invertedTestStats.get(metric));
                String keyOverallMed = _calcMedianStringFromStringList(invertedTestStats.get(metric));

                writer.println("s: " + keyOverallStd + " (std)");
                writer.println("m: " + keyOverallMed + " (median)\n");
                writer.println("a: " + keyOverallAvg + " (average)");

                for(int fld = 1; fld <= numFolds; fld++) {
                    String fldStr = String.valueOf(fld);
                    String currVal = val.get(fldStr);
                    writer.println(fldStr + ": " + currVal);
                }
                writer.print("\n");
            }
            writer.println("---- Test Fold-level Statistics End ----");
            writer.println("\n");

            writer.println("---- Develop Fold-level Statistics Begin ----");

            writer.println("- This module provides statistics for each fold");
            writer.println("- averaged over all entity types\n");

            for (HashMap.Entry<String, Map<String, String>> entry : foldLevelDevelopStats.entrySet()) {
                String metric = entry.getKey();
                Map<String, String> val = entry.getValue();

                writer.println("# " + metric);
                String keyOverallAvg = avgDevelopStats.get(metric).get("overall");
                String keyOverallStd = _calcStdStringFromStringList(invertedDevelopStats.get(metric));
                String keyOverallMed = _calcMedianStringFromStringList(invertedDevelopStats.get(metric));

                writer.println("s: " + keyOverallStd + " (std)");
                writer.println("m: " + keyOverallMed + " (median)\n");
                writer.println("a: " + keyOverallAvg + " (average)");

                for(int fld = 1; fld <= numFolds; fld++) {
                    String fldStr = String.valueOf(fld);
                    String currVal = val.get(fldStr);
                    writer.println(fldStr + ": " + currVal);
                }
                writer.print("\n");
            }
            writer.println("---- Develop Fold-level Statistics End ----");
            writer.println("\n");

            writer.close();
        } catch (IOException e) {
            logger.severe(e.getMessage());
        }
    }

    private void writeExperimentDurationToFile(String fldStr, HashMap<String, String> fldRunData){
        logger.info("writing experiment's duration to file...");

        String fileName = outputFilesCvDatasetDatetimeDir + "/durations.txt";

        BufferedWriter bw = null;
        FileWriter fw = null;

        try {
            long secondsDiff = _calcExperimentDurationFromRunData(fldRunData);
            long minutesDiff = secondsDiff / 60;

            String minutesDiffStr = String.valueOf(minutesDiff);
            String secondsDiffStr = String.valueOf(secondsDiff);

            String lineToAppend = "\nfold #" + fldStr + ": " + minutesDiffStr + "m " + "(" + secondsDiffStr + ")s";

            File file = new File(fileName);

            // if file doesn't exist, create it
            if (!file.exists()) {
                file.createNewFile();
            }

            // true = append file
            fw = new FileWriter(file.getAbsoluteFile(), true);
            bw = new BufferedWriter(fw);

            bw.write(lineToAppend);

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null)
                    bw.close();
                if (fw != null)
                    fw.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    private void writeCrashDuringCVToFile(String content){
        String fileName = outputFilesCvDatasetDatetimeDir + "/crash.txt";

        File directory = new File(outputFilesCvDatasetDatetimeDir);
        if (!directory.exists()) {
            directory.mkdir();
        }

        try{
            PrintWriter writer = new PrintWriter(fileName, "UTF-8");
            writer.println("---- Begin Datetime ----");
            writer.println(cvBeginDateTime);
            writer.println("\n");
            writer.println("---- run crashed during CV ----");
            writer.println(content);
            writer.close();

        } catch (IOException e) {
            logger.severe(e.getMessage());
        }
    }

    private String _getOutputFileName(String type){
        HashMap<String, String> runData = this.runParams;
        HashMap<String, Map<String, String>> testStats = this.avgTestStats;
        HashMap<String, Map<String, String>> developStats = this.avgDevelopStats;

        // long filename
        String fileNameInfo = cvBeginDateTime + "__numiter=" + runData.get("numTrainIterations") +
                "__testAccuracy=" + String.format("%.3f", Double.valueOf(testStats.get("Accuracy").get("overall"))) + "__testF1=" + String.format("%.3f", Double.valueOf(testStats.get("F1").get("overall"))) +
                "__developAccuracy=" + String.format("%.3f", Double.valueOf(developStats.get("Accuracy").get("overall"))) + "__developF1=" + String.format("%.3f", Double.valueOf(developStats.get("F1").get("overall")));


        if (type.equals("avg")){
            fileNameInfo = "__avg" + String.valueOf(completedFoldsCounter) + "__" + fileNameInfo;
        } else if (type.equals("fld")) {
            fileNameInfo = "__fld" + String.valueOf(completedFoldsCounter) + "__" + fileNameInfo;
        } else if (type.equals("sentence")) {
            fileNameInfo = "__sentence-level" + String.valueOf(completedFoldsCounter);
        }

        // short filename
        //		String fileNameInfo = date + "__" + updatorClassStr;

        return fileNameInfo;
    }

    private String _calcStdStringFromStringList(List<String> sample){
        double average = 0.0;
        double std = 0.0;

        for(String a : sample)
            average += Double.valueOf(a);
        average = average/sample.size();

        for(String a : sample){
            Double aStr = Double.valueOf(a);
            std += (aStr - average)*(aStr - average);
        }

        // save division (on first fold we divide by zero)
        int sampleSize = sample.size();
        if (sampleSize > 0) {
            std = Math.sqrt(std / (sampleSize - 1));
        } else {
            std = 0;
        }

        return String.valueOf(std);
    }

    private String _calcMedianStringFromStringList(List<String> sample){
        double[] data = new double[sample.size()];

        for(int i=0; i<sample.size(); i++){
            data[i] = Double.valueOf(sample.get(i));
        }

        Arrays.sort(data);

        double median = 0.0;

        if (data.length % 2 == 0) {
            median = (data[(data.length / 2) - 1] + data[data.length / 2]) / 2.0;
        } else {
            median = data[data.length / 2];
        }
        return String.valueOf(median);
    }

    private long _calcExperimentDurationFromRunData(HashMap<String, String> fldRunData) {
        long secondsDiff = 0;
        try {
            String beganExperimentDatetime = fldRunData.get("began experiment datetime");
            String finishExperimentDatetime = fldRunData.get("finish experiment datetime");

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

    private HashMap<String, Boolean> _getFeatureMap(){
        HashMap<String, Boolean> featureMap = new HashMap<>();

        /// unary
        featureMap.put("w[i]", true);
        featureMap.put("w[i-1]", true);
        featureMap.put("w[i+1]", true);

        featureMap.put("t[i]", true);
        featureMap.put("t[i-1]", true);
        featureMap.put("t[i+1]", true);

        /// binary
        featureMap.put("t[i-1], t[i]", true);
        featureMap.put("t[i], t[i+1]", true);
        featureMap.put("t[i-1], t[i+1]", true);

        featureMap.put("w[i], w[i-1]", true);
        featureMap.put("w[i], w[i+1]", true);
        featureMap.put("w[i-1], w[i+1]", true);

        featureMap.put("w[i], t[i-1]", true);
        featureMap.put("w[i], t[i]", true);
        featureMap.put("w[i], t[i+1]", true);

        /// ternary
        featureMap.put("t[i-2], t[i-1], t[i]", true);
        featureMap.put("w[i], t[i-1], t[i]", true);
        featureMap.put("w[i], t[i-1], t[i+1]", true);
        featureMap.put("w[i-1], t[i], t[i+1]", true);

        return featureMap;
    }

    public void run(){
        logger.info("\n------- Cross-Validation began -------");
        validateFiles();
        printCvParams();
        performCrossValidation();
        processResults();
        processEntityLevelResults();
        writeAvgResultsToFile();
        writeDetailedResultsToFile();
        logger.info("\n------- Cross-Validation completed -------");
    }

}


