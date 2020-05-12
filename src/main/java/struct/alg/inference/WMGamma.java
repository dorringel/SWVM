package struct.alg.inference;

import java.util.List;

import struct.alg.Predictor;
import struct.sequence.SequenceInstance;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;

public class WMGamma implements GammaCalculator {
	SequenceInstance goldInst;
	double beta;
	public double sum;
	Predictor pred;

	/**
	 * Create an aggressive weighted margin Gamma object.
	 *
	 * @throws IllegalArgumentException
	 *             if one of the JJs is not a violation w.r.t. the weights and
	 *             the gold
	 */
	public WMGamma(SequenceInstance goldInst, List<SLLabel> JJs, double beta, Predictor pred)
			throws IllegalArgumentException {
		this.goldInst = goldInst;
		this.beta = beta;
		this.beta=beta;
		this.sum = 1.0;
		this.pred = pred;
		
		
		//Calculate the sum of the margins.
		SLFeatureVector gold_fv = goldInst.getLabel().getFeatureVectorRepresentation(); // gold
		for (SLLabel mj : JJs) {
			SLFeatureVector mj_fv = mj.getFeatureVectorRepresentation();
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);
			double delta = dist_fv.dotProdoct(pred.weights);
			this.sum += Math.pow(Math.abs(1 + delta), this.beta);

		}

	}

	@Override
	public double calculate(SLLabel mJ) {
		SLFeatureVector mj_fv = mJ.getFeatureVectorRepresentation();
		SLFeatureVector gold_fv = goldInst.getLabel().getFeatureVectorRepresentation();
		SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, mj_fv);
		double gamma = Math.abs(dist_fv.dotProdoct(pred.weights));
		return (this.sum == 0 ? 1 : Math.pow(1 + gamma / this.sum, this.beta));
	}
}
