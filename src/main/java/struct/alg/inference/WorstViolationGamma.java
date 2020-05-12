package struct.alg.inference;

import java.util.ArrayList;
import java.util.List;

import gnu.trove.TIntDoubleHashMap;
import struct.alg.Predictor;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;

public class WorstViolationGamma implements GammaCalculator {
	int worstIndex;
	List<SLLabel> ma;
	
	public WorstViolationGamma(SLFeatureVector gold_fv, SLFeatureVector pred_fv ,List<SLLabel> JJs, Predictor pred) {
		this.ma = JJs;
		ArrayList<double[]> weightsPerJJ = new ArrayList<>();
		SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, pred_fv);
		int maxIndex = SLFeatureVector.getMaxIndex(dist_fv);
		//Per mixed assignment, calculate what the weights would be if we gave only that mixed assignment gamma=1 and the rest gamma=0
		for(int i=0; i<JJs.size(); i++){
			double[] weights = new double[maxIndex+1];
			TIntDoubleHashMap dist_hm = SLFeatureVector.Concentrate(SLFeatureVector.getDistVector(gold_fv, JJs.get(i).getFeatureVectorRepresentation()));
			for(int j = 0; j < weights.length; j++){
				if(dist_hm.containsKey(j))
					weights[j]=pred.weights[j]+dist_hm.get(j);
				else
					weights[j]=pred.weights[j];
			}
			weightsPerJJ.add(i, weights);
		}
		
		double[] weightDots = new double[weightsPerJJ.size()];
		for(int i=0; i<weightDots.length; i++){
			weightDots[i] = dist_fv.dotProdoct(weightsPerJJ.get(i));
		}
		
		//This is an argmin function.
		double worstval = Double.MAX_VALUE;
		for(int i=0; i<weightDots.length; i++){
			if(weightDots[i]<worstval){
				worstval = weightDots[i];
				worstIndex = i;
			}
				
		}
		
	}
	
	@Override
	public double calculate(SLLabel mJ) {
		if(this.ma.get(this.worstIndex) == mJ)
			return 1.0;
		else
			return 0;
	}

}
