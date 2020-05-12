package struct.alg;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import gurobi.GRB;
import struct.alg.clustering.MAClusterer;
import struct.alg.inference.GammaCalculator;
import struct.alg.inference.OptimizedGamma;
import struct.alg.inference.MixedAssignmentSeperator;
import struct.alg.inference.StatisticalGamma;
import struct.sequence.MACreator;
import struct.sequence.SequenceFeatures;
import struct.sequence.SequenceInstance;
import struct.sequence.SequencePrediction;
import struct.types.*;
import weka.clusterers.HierarchicalClusterer;

/**
 * Updates Perceptron using two phase methods: (1) finding all JJs - a set of
 * tags for which the a single tag is taken from the predicted and the rest from
 * the gold label (2) Adjusting weights according to Gamma.
 */
public class SWVPUpdator implements OnlineUpdator {

	public static int ctr = 0;
	private MACreator creator;
	// deprecated begin
	private boolean clusterMAsFlag;
	// deprecated end
	private int topK;
	private String maType = "";
	private String jjType;

	private String gammaCalculationMethod;
	private double gammaCalculationBeta = 1;
	private int gammaCalculationObjectiveType = GRB.MINIMIZE;

	public SWVPUpdator(){}

	public SWVPUpdator(boolean clusterMAsFlag, int topK, String maType, String jjType,
					   String gammaCalculationMethod, double gammaCalculationBeta, String gammaCalculationObjectiveType) {
		try {
			this.clusterMAsFlag = clusterMAsFlag;
			this.topK = topK;
			this.maType = maType;
			this.jjType = jjType;

			this.gammaCalculationMethod = gammaCalculationMethod;		// uniform, wm, wmr, softmax, softmin, one, opt
			this.gammaCalculationBeta = gammaCalculationBeta;			// [0.1,10]

			if (gammaCalculationObjectiveType.equals("MINIMIZE")) { this.gammaCalculationObjectiveType = GRB.MINIMIZE; }
			else if (gammaCalculationObjectiveType.equals("MAXIMIZE")) { this.gammaCalculationObjectiveType = GRB.MAXIMIZE; }

		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}


	/**
	 * This method performs the update rule in respect to
	 * the top-K (of best-K) structures received by inference
	 * it basically run the regular update routine in a loop K times
	 * and can certainly be implemented in a more efficient way
	 * sending K=1 is similar to the regular update of using the best prediction
	 *
	 * @param inst
	 * @param feats
	 * @param predictor
	 * @param avg_upd
	 */
	@Override
	public void update(SLInstance inst, Features feats, Predictor predictor, double avg_upd) {
		ctr++;
		if (ctr >= 50) {
			// System.out.println("GC!");
			ctr = 0;
			System.gc();
		}

		creator = new MACreator((SequenceFeatures) feats);
		Prediction pred = predictor.decode(inst, feats, this.topK); // inference
		SequencePrediction sqPred = null;
		SequenceInstance goldInst = null;

		// downcast
		if ((inst instanceof SequenceInstance) && (pred instanceof SequencePrediction)) {
			sqPred = (SequencePrediction) pred;
			goldInst = (SequenceInstance) inst;
		} else {
			throw new IllegalArgumentException("can't iterate over prediction sentence");
		}

		// y (gold label)
		SLFeatureVector gold_fv = goldInst.getLabel().getFeatureVectorRepresentation();

		SLLabel[] labels = new SLLabel[topK];

		// y* (label from prediction)

		// Create mixed assignment objects
		// Run the regular update routine K times in a loop and perform update each time
		for (int j = 0; j < this.topK; j++) {

			// y*_j (label from prediction)
			labels[j] = sqPred.getLabelByRank(j);
			SLFeatureVector pred_fv = sqPred.getLabelByRank(j).getFeatureVectorRepresentation();

			Map<SLLabel, List<Integer>> maMap;
			if (this.jjType.equals("single")) {
				maMap = getSingleMixedAssignments(goldInst, sqPred, j);
			} else if (this.jjType.equals("double")) {
				maMap = getDoubleMixedAssignments(goldInst, sqPred, j);
			} else { // else if (this.jjType.equals("triple")) {
				maMap = getTripleMixedAssignments(goldInst, sqPred, j);
			}

			if (clusterMAsFlag) {
				try {
					maMap.putAll(new MAClusterer(new HierarchicalClusterer()).cluster(maMap, new String[]{}, creator));// "-N",
					//"7"
				} catch (Exception e) {
					e.printStackTrace();
					throw new IllegalArgumentException("can't cluster Mixed Assignments");
				}
			}

			// Separate the mixed assignments into passive and aggressive.
			MixedAssignmentSeperator sep = new MixedAssignmentSeperator(predictor,
					goldInst.getLabel().getFeatureVectorRepresentation(), maMap.keySet());

			List<SLLabel> final_ma = sep.passive;		// default initialization
			if (this.maType.equals("passive")) {
				final_ma = sep.passive;
			} else if (this.maType.equals("aggressive")) {
				final_ma = sep.aggressive;
			} else if (this.maType.equals("all")){
				final_ma = sep.aggressive;
				final_ma.addAll(sep.passive);
			} else{ }

			int maSize = final_ma.size();
			if (maSize == 0){
				final_ma.addAll(sep.passive);
			}


			double[] gammaArray = null;
			HashMap<SLLabel, Double> maToGammaMap = null;

			StatisticalGamma statisticalGammaObj;
			OptimizedGamma OptimizedGammaObj;


			try {

				SLFeatureVector weights = null;

				// TODO: change if/else to reflection

				if (gammaCalculationMethod.equals("uniform")) {
					statisticalGammaObj = new StatisticalGamma(gold_fv, pred_fv, final_ma, predictor);
					maToGammaMap = statisticalGammaObj.calcUniformGamma();
					gammaArray = statisticalGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, statisticalGammaObj);

				} else if (gammaCalculationMethod.equals("wm")) {
					statisticalGammaObj = new StatisticalGamma(gold_fv, pred_fv, final_ma, predictor);
					maToGammaMap = statisticalGammaObj.calcWmGamma(gammaCalculationBeta);
					gammaArray = statisticalGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, statisticalGammaObj);

				} else if (gammaCalculationMethod.equals("wmr")) {
					statisticalGammaObj = new StatisticalGamma(gold_fv, pred_fv, final_ma, predictor);
					maToGammaMap = statisticalGammaObj.calcWmrGamma(gammaCalculationBeta);
					gammaArray = statisticalGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, statisticalGammaObj);

				} else if (gammaCalculationMethod.equals("softmax")) {
					statisticalGammaObj = new StatisticalGamma(gold_fv, pred_fv, final_ma, predictor);
					maToGammaMap = statisticalGammaObj.calcSoftmaxGamma();
					gammaArray = statisticalGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, statisticalGammaObj);

				} else if (gammaCalculationMethod.equals("softmin")) {
					statisticalGammaObj = new StatisticalGamma(gold_fv, pred_fv, final_ma, predictor);
					maToGammaMap = statisticalGammaObj.calcSoftminGamma();
					gammaArray = statisticalGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, statisticalGammaObj);

				} else if (gammaCalculationMethod.equals("one")) {
					OptimizedGammaObj = new OptimizedGamma(gold_fv, pred_fv, final_ma, predictor, this.gammaCalculationObjectiveType);
					OptimizedGammaObj.optimizationOne();
					gammaArray = OptimizedGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, OptimizedGammaObj);

				} else if (gammaCalculationMethod.equals("opt")) {
					OptimizedGammaObj = new OptimizedGamma(gold_fv, pred_fv, final_ma, predictor, this.gammaCalculationObjectiveType);
					OptimizedGammaObj.optimizationFour();
					gammaArray = OptimizedGammaObj.getGammaArray();
					weights = calcUpdateToWeights(gold_fv, predictor,
							final_ma, OptimizedGammaObj);

				}

				updateWeights(predictor, avg_upd, weights);

			} catch (Exception e){
				System.out.println(e.getMessage());
				System.exit(1);
			}



		}
	}

	/**
	 * This is a simple weight updater to the weights in the predicater. For
	 * every update in calcWeights, add the update to the weights at the same
	 * index.
	 *
	 * @param predictor
	 * @param avg_upd
	 * @param calcWeights
	 */
	private void updateWeights(Predictor predictor, double avg_upd, SLFeatureVector calcWeights) {
		double[] weights = predictor.weights;
		double[] avg_weights = predictor.avg_weights;
		// This method is from SLFeatureVector to iterate over feature vectors
		// and add them to weight arrays.
		for (SLFeatureVector curr = calcWeights; curr != null; curr = curr.next) {
			if (curr.index >= 0) {
				weights[curr.index] += curr.value;
				avg_weights[curr.index] += avg_upd * curr.value;
			}
		}
	}


	/**
	 * Creates list of mixed assignments including single tag from the predicted
	 * sequence (all other tags are from the gold lable). The feature vector is
	 * also calculated for each element in this list
	 * >> an additional argument for the rank of the prediction (used for Top-K updates)
	 * @return a list of labels according to the size of input instance
	 */
	Map<SLLabel, List<Integer>> getSingleMixedAssignments(SequenceInstance goldInst, SequencePrediction sqPred, int rank) {
		Map<SLLabel, List<Integer>> $ = new HashMap<>(sqPred.getNumLabels());

		// Y and Y*
		String[] goldTags = goldInst.getLabel().tags, predTags = sqPred.getLabelByRank(rank).tags;

		// for K best predictions you'll need
		// Arrays.copyOfRange(labels, 0, K)

		for (int i = 0; i < predTags.length; ++i) {
			// if(goldTags[i].compareTo(predTags[i]) == 0)
			// continue;
			String[] jj = getMA(goldTags, predTags, i);
			// get feature vector for a given JJ
			List<Integer> indices = new LinkedList<>();
			indices.add(i);
			$.put(creator.create(jj/* , goldInst.getInput().sentence.length */), indices);
		}
		return $;
	}

	Map<SLLabel, List<Integer>> getSingleMixedAssignments(SequenceInstance goldInst, SequencePrediction sqPred){
		return getSingleMixedAssignments(goldInst, sqPred, 0);
	}

	/**
	 * Creates list of mixed assignments including two tags from the predicted
	 * sequence (all other tags are from the gold label). The feature vector is
	 * also calculated for each element in this list
	 *
	 * @return a list of labels according to the size of input instance
	 */
	Map<SLLabel, List<Integer>> getDoubleMixedAssignments(SequenceInstance goldInst, SequencePrediction sqPred, int rank) {
		//List<SLLabel> $ = new ArrayList<>(sqPred.getNumLabels());
		Map<SLLabel, List<Integer>> $ = new HashMap<>();

		// Y and Y*
		String[] goldTags = goldInst.getLabel().tags, predTags = sqPred.getLabelByRank(rank).tags;
		// MixedAssignmentCreator creator = new MixedAssignmentCreator(sf);

		for (int i = 0; i < predTags.length; ++i) {
			if (goldTags[i].equals(predTags[i]))
				continue;
			for (int j = i; j < predTags.length; ++j) {
				if (i != j) {
					if (goldTags[j].equals(predTags[j]))
						continue;
					String[] jj = Arrays.copyOf(goldTags, goldTags.length);
					jj[i] = predTags[i];
					jj[j] = predTags[j];

					// create indices for HashMap value
					List<Integer> indices = new LinkedList<>();
					indices.add(i);
					indices.add(j);

					// get feature vector for a given JJ
					$.put(creator.create(jj/* , goldInst.getInput().sentence.length */), indices);
				}
			}
		}
		return $;
	}

	/**
	 * Creates list of mixed assignments including three tags from the predicted
	 * sequence (all other tags are from the gold label). The feature vector is
	 * also calculated for each element in this list
	 *
	 * @return a list of labels according to the size of input instance
	 */
	Map<SLLabel, List<Integer>> getTripleMixedAssignments(SequenceInstance goldInst, SequencePrediction sqPred, int rank) {
		//List<SLLabel> $ = new ArrayList<>(sqPred.getNumLabels());
		Map<SLLabel, List<Integer>> $ = new HashMap<>();


		// Y and Y*
		String[] goldTags = goldInst.getLabel().tags, predTags = sqPred.getLabelByRank(rank).tags;
		// MixedAssignmentCreator creator = new MixedAssignmentCreator(sf);

		for (int i = 0; i < predTags.length; ++i) {
			for (int j = i; j < predTags.length; ++j) {
				for (int k = j; k < predTags.length; ++k) {
					if (i != j && i != k && j != k) {
						if (goldTags[i].compareTo(predTags[i]) == 0 || goldTags[j].compareTo(predTags[j]) == 0
								|| goldTags[k].compareTo(predTags[k]) == 0)
							continue;
						String[] jj = Arrays.copyOf(goldTags, goldTags.length);
						jj[i] = predTags[i];
						jj[j] = predTags[j];
						jj[k] = predTags[k];

						// create indices for HashMap value
						List<Integer> indices = new LinkedList<>();
						indices.add(i);
						indices.add(j);
						indices.add(k);

						// get feature vector for a given JJ
						$.put(creator.create(jj/* , goldInst.getInput().sentence.length */), indices);
					}

				}
			}
		}
		return $;
	}

	/**
	 * @return tags array composed of the gold label in all positions besides
	 *         the i-th position that is of the predicted tags
	 */
	String[] getMA(String[] goldTags, String[] predTags, int i) {
		String[] $ = Arrays.copyOf(goldTags, goldTags.length);
		$[i] = predTags[i];
		return $;
	}

	/**
	 * Calculate the updates to the weights given mixed assignments and a gamma
	 * function
	 *
	 * @param gold_fv
	 * @param predictor
	 * @param final_ma
	 * @param gamma_calc
	 * @return
	 */
	private SLFeatureVector calcUpdateToWeights(SLFeatureVector gold_fv, Predictor predictor, List<SLLabel> final_ma,
			GammaCalculator gamma_calc) {
		// This feature vector will actually contain the weight updates per
		// index in the weight vector
		SLFeatureVector $ = new SLFeatureVector(-1, -1.0, null);

		// For every mixed assignment, create a distance vector phi(x,y) -
		// phi(x,y') and calculate the gamma
		for (SLLabel mJ : final_ma) {

			SLFeatureVector mj_fv = mJ.getFeatureVectorRepresentation();
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);

			double mj_gamma = gamma_calc.calculate(mJ);

			// We normalize by 1/gamma because the normalization divides by
			// gamma, so we actually multiply like we want
			$.addCat(SLFeatureVector.normalize(dist_fv, 1 / mj_gamma));

		}

		return $;
	}


}
