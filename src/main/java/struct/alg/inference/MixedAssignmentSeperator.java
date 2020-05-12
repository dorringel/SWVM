package struct.alg.inference;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import struct.alg.Predictor;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;

public class MixedAssignmentSeperator {
	public List<SLLabel> aggressive;
	public List<SLLabel> passive;

	// public MixedAssignmentSeperator(Predictor pred, SequenceInstance
	// goldInst, List<SLLabel> mixedAssignments) {
	// this.aggressive = new LinkedList<SLLabel>();
	// this.passive = new LinkedList<SLLabel>();
	// SLFeatureVector gold_fv =
	// goldInst.getLabel().getFeatureVectorRepresentation(); // gold
	// for(SLLabel ma : mixedAssignments){
	// SLFeatureVector ma_fv = ma.getFeatureVectorRepresentation();
	// SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, ma_fv);
	// if (dist_fv.dotProdoct(pred.weights) <= 0) {
	// this.aggressive.add(ma);
	// } else {
	// this.passive.add(ma);
	// }
	// }
	// }

	public MixedAssignmentSeperator(Predictor pred, SLFeatureVector gold_fv, Set<SLLabel> set) {
		this.aggressive = new LinkedList<>();
		this.passive = new LinkedList<>();
		for (SLLabel ma : set) {
			SLFeatureVector ma_fv = ma.getFeatureVectorRepresentation();
			SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, ma_fv);
			if (dist_fv.dotProdoct(pred.weights) <= 0) {
				this.aggressive.add(ma);
			} else {
				this.passive.add(ma);
			}
		}
		// //Ascending order
		// this.aggressive.sort(new Comparator<SLLabel>() {
		//
		// @Override
		// public int compare(SLLabel o1, SLLabel o2) {
		// SLFeatureVector o1fv = o1.getFeatureVectorRepresentation();
		// SLFeatureVector o2fv = o2.getFeatureVectorRepresentation();
		// SLFeatureVector d1fv = SLFeatureVector.getDistVector(gold_fv, o1fv);
		// SLFeatureVector d2fv = SLFeatureVector.getDistVector(gold_fv, o2fv);
		// double $ = d1fv.dotProdoct(pred.weights) -
		// d2fv.dotProdoct(pred.weights);
		// if($==0)
		// return 0;
		// else
		// return ($>0? 1 : -1);
		// }
		// });

		// List<SLLabel> filtered_agg = new ArrayList<>();
		// for(int i=0; i<50; i++){
		// if(i>=this.aggressive.size())
		// break;
		// filtered_agg.add(this.aggressive.get(i));
		// }
		// this.aggressive = filtered_agg;
	}

	public List<SLLabel> getBalancedMAs() {
		return null;
	}
}
