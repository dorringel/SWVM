package examples;

import java.util.HashMap;

public class Main {

    public static void main(String[] args) {
        HashMap<String, String> cvParams = new HashMap<>();

        // ------------------------------------------------------------------------------------------------------- //
        // ------------------------------------------------------------------------------------------------------- //
        // ------------------------------------------------------------------------------------------------------- //
        // ------------------------------------------------------------------------------------------------------- //
        // ---------- CONFIGURABLE SETTING BEGIN HERE ------------------------------------------------------------ //

        // -- dataset to run -- //

        /* currently supporting the following:

        genia, genia_small, genia_standard
        bc2gm, bc2gm_standard
        conll2002, conll2002_standard
        conll2000-chunking, conll2000-chunking_small_025, conll2000-chunking_standard
        conll2007-pos, conll2007-pos_small_05, conll2007-pos_small_025, conll2007-pos_standard
        conll2007-pos_small_01, conll2007-pos_small_001
        genia-pos, genia-pos_small_025
        synthetic3, synthetic4, synthetic5"

         */

        cvParams.put("dataset", "genia");

        // -- number of iterations to run (15 has been found as 'best') --//
        cvParams.put("numTrainIterations", "15");

        // -- number of folds to run (maximum is 9) -- //
        cvParams.put("maxFolds", "5");

        // -- non-negative gamma -- //
        // OptimizedGamma - line 305
        /// Add constraint: gamma[i]>=0 for all i
        // this does nothing except for reminding you that this option is on
        // you need to manually uncomment the relevant code section
        //cvParams.put("non-negative-gammas", "non-negative-gammas");

        // -- updator class and updator parameters -- //

//        /// CSP
        cvParams.put("updatorClass", "PerceptronUpdator");


        /// MIRA
//        cvParams.put("updatorClass", "KBestMiraUpdator");
//        cvParams.put("kMira", "7");

        /// SWVP
//        cvParams.put("updatorClass", "SWVPUpdator");
//        cvParams.put("topK", "1");                          // update with respect to topK inference results
//        cvParams.put("maType", "aggressive");                      // aggressive/passive/all mixed assignments
//        cvParams.put("jjType", "single");                   // single/double mixed assignments
//        cvParams.put("gammaCalculationMethod", "softmin");     // uniform/wm/wmr/softmax/softmin/one/opt methods to calculating gammas
//        cvParams.put("gammaCalculationBeta", "1");          // [0.1,10] beta values for gamma calculation
//        cvParams.put("gammaCalculationObjectiveType", "MAXIMIZE");     // MINIMIZE/MAXIMIZE objectives


        /// SWVM
//        cvParams.put("updatorClass", "SWVMUpdator");
//        cvParams.put("topK", "1");                          // update with respect to topK inference results
//        cvParams.put("maType", "aggressive");                      // aggressive/passive/all mixed assignments
//        cvParams.put("jjType", "single");                   // single/double mixed assignments
//        cvParams.put("gammaCalculationMethod", "wm");     // uniform/wm/wmr/softmax/softmin/one/opt methods to calculating gammas
//        cvParams.put("gammaCalculationBeta", "1");          // [0.1,10] beta values for gamma calculation
//        cvParams.put("gammaCalculationObjectiveType", "MAXIMIZE");     // MINIMIZE/MAXIMIZE objectives



        // ---------- CONFIGURABLE SETTING END HERE ------------------------------------------------------------ //
        // ------------------------------------------------------------------------------------------------------- //
        // ------------------------------------------------------------------------------------------------------- //
        // ------------------------------------------------------------------------------------------------------- //
        // ------------------------------------------------------------------------------------------------------- //

        CrossValidation cv = new CrossValidation(cvParams);

        cv.run();
    }
}
