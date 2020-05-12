package struct.alg.inference;
import struct.alg.Predictor;
import struct.types.ArrayIndexComparator;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;

import java.util.*;

public class StatisticalGamma implements GammaCalculator {
	double[] gammaArray;
	HashMap<SLLabel, Double> maToGammaMap;
	//static GRBEnv env = null;
	//boolean didSolve = true;

	private SLFeatureVector gold_fv;
	private SLFeatureVector pred_fv;
	private List<SLLabel> JJs;
	private Predictor pred;
	private int maSize;

	/**
	 *
	 * @param gold_fv - true output [sequence linear feature vector]
	 * @param pred_fv - predicted output [sequence linear feature vector]
	 * @param JJs - mixed-assignments [linked list]
	 * @param pred - predictor (which contains W) [sequence predictor]
	 */
	public StatisticalGamma(SLFeatureVector gold_fv, SLFeatureVector pred_fv , List<SLLabel> JJs, Predictor pred) {
			this.gold_fv = gold_fv;
			this.pred_fv = pred_fv;
			this.JJs = JJs;
			this.pred = pred;
			this.maSize = JJs.size();
			this.gammaArray = new double[maSize];
			this.maToGammaMap = new HashMap<>();
	}

	/**
	 * @return maToGammaMap representing the array for gammas.
	 * calculate gammas uniformly
	 */
	public HashMap<SLLabel, Double> calcUniformGamma(){
		double currGamma = 1/(double)maSize;

		for (int i=0; i<maSize; i++) {
			gammaArray[i] = currGamma;
			maToGammaMap.put(JJs.get(i), currGamma);
		}

		return maToGammaMap;
	}

	/**
	 * @return maToGammaMap representing the array for gammas.
	 * calculate gammas using Weighted Margin (WM) method from the SWVP article
	 */
	public HashMap<SLLabel, Double> calcWmGamma(double beta){
		double[] deltaArray = new double[maSize];
		double sum = 0;

		// calculate |w * delta-phi(x, y, m^J)| ^ beta for each m^J then normalize
		int i = 0;
		for (SLLabel mj : JJs) {

			// get feature vectors
			SLFeatureVector mj_fv = mj.getFeatureVectorRepresentation();
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);

			// calculate w * delta-phi dot product
			double delta = dist_fv.dotProdoct(pred.weights);

			// take absolute value
			double delta_abs = Math.abs(1 + delta);

			// raise to the power of beta
			double delta_final = Math.pow(delta_abs, beta);

			// assign current delta and cumulative sum
			deltaArray[i] = delta_final;
			sum += delta_final;

			i++;
		}

		// normalize and assign gammas
		for (i=0; i<maSize; i++) {
			double currGamma = deltaArray[i] / sum;
			gammaArray[i] = currGamma;
			maToGammaMap.put(JJs.get(i), currGamma);
		}

		return maToGammaMap;
	}

	/**
	 * @return maToGammaMap representing the array for gammas.
	 * calculate gammas using Weighted Margin Rank (WMR) method from the SWVP article
	 */
	public HashMap<SLLabel, Double> calcWmrGamma(double beta){
		double[] deltaArray = new double[maSize];

		// calculate |w * delta-phi(x, y, m^J)| for each m^J
		int i = 0;
		for (SLLabel mj : JJs) {

			// get feature vectors
			SLFeatureVector mj_fv = mj.getFeatureVectorRepresentation();
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);

			// calculate w * delta-phi dot product
			double delta = dist_fv.dotProdoct(pred.weights);

			// take absolute value
			double delta_final = Math.abs(1 + delta);

			// assign current delta
			deltaArray[i] = delta_final;

			i++;
		}

		// build indices array
		ArrayIndexComparator comparator = new ArrayIndexComparator(deltaArray);
		Integer[] indicesArray = comparator.createIndexArray();

		// sort indices array from smallest to largest
		Arrays.sort(indicesArray, comparator);

		// the sum of the numerators is (|JJ_x| - 1)/2
		// thus there is not need to calculate it as we loop
		double sum = maSize > 1 ? (double)(maSize - 1) / 2 : 1.0;

		for (i=0; i<maSize; i++){

			// calculate current gamma value
			double base = (double)(maSize - (i+1)) / maSize;
			double powered = Math.pow(base, beta);

			// normalize and assign gammas
			double currGamma = powered / sum;
			gammaArray[indicesArray[maSize-1-i]] = currGamma;
			maToGammaMap.put(JJs.get(indicesArray[maSize-1-i]), currGamma);
		}

		return maToGammaMap;
	}

	/**
	 * reverses an array in place
	 * @param array of Integers
	 * @return void
	 */
	void reverseIntegerArrayInPlace(Integer[] array) {
		for (int i = 0; i < array.length / 2; i++) {
			int temp = array[i];
			array[i] = array[array.length - i - 1];
			array[array.length - i - 1] = temp;
		}
	}


	/**
	 * @return maToGammaMap representing the array for gammas.
	 * calculate gammas using Softmax calculation of the various substructures
	 */
	public HashMap<SLLabel, Double> calcSoftmaxGamma(){
		double[] deltaArray = new double[maSize];
		double exponentialSum = 0;

		//Calculate the exponential sum of the margins.
		int i = 0;
		for (SLLabel mj : JJs) {
			SLFeatureVector mj_fv = mj.getFeatureVectorRepresentation();

			// calc gold_fv - mj_fv L1-norm [phi(x,y) - phi(x,m_j)]
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);

			// calc w * phi(x,y) - phi(x,m_j) = w * phi(x,y,m_j)
			double delta = dist_fv.dotProdoct(pred.weights);

			// negative exponent of delta (high negative delta means high violation)s
			double delta_exp = Math.exp(delta);
			deltaArray[i] = delta_exp;
			exponentialSum += delta_exp;
			i++;
		}

		// create array of gammas
		for (i=0; i<maSize; i++){
			double currGamma = deltaArray[i] / exponentialSum;
			gammaArray[i] = currGamma;
			maToGammaMap.put(JJs.get(i), currGamma);
		}

		return maToGammaMap;
	}

	/**
	 * @return maToGammaMap representing the array for gammas.
	 * calculate gaammas using Softmin calculation of the various substructures
	 */
	public HashMap<SLLabel, Double> calcSoftminGamma(){
		double[] deltaArray = new double[maSize];
		double exponentialNegativeSum = 0;

		//Calculate the negative exponential average of the margins (softmin).
		int i = 0;
		for (SLLabel mj : JJs) {
			SLFeatureVector mj_fv = mj.getFeatureVectorRepresentation();

			// calc gold_fv - mj_fv L1-norm [phi(x,y) - phi(x,m_j)]
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);

			// calc w * phi(x,y) - phi(x,m_j) = w * phi(x,y,m_j)
			double delta = dist_fv.dotProdoct(pred.weights);

			// negative exponent of delta (high negative delta means high violation)s
			double delta_exp = Math.exp(-delta);
			deltaArray[i] = delta_exp;
			exponentialNegativeSum += delta_exp;
			i++;
		}

		// create array of gammas
		for (i=0; i<maSize; i++){
			double currGamma = deltaArray[i] / exponentialNegativeSum;
			gammaArray[i] = currGamma;
			maToGammaMap.put(JJs.get(i), currGamma);
		}

		return maToGammaMap;
	}



	@Override
	public double calculate(SLLabel mJ) {
		return maToGammaMap.get(mJ);
	}

	public double[] getGammaArray(){
		return this.gammaArray;
	}

}
