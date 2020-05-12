package struct.sequence;

import java.util.List;

import cc.mallet.types.Alphabet;
import struct.types.SLFeatureVector;

public class MACreator {

	private final SequenceFeatures sf;

	// public JJSequencDecoder(SequencePredictor predictor) {
	// this.predictor = predictor;
	// }

	public MACreator(SequenceFeatures sf) {
		this.sf = sf;
	}

	/**
	 * decodes input tags to a feature vector
	 *
	 * @return SequenceLabel including the tags and the features
	 */
	public SequenceLabel create(String[] tags/* , int instLength */) {
		Alphabet tagAlphabet = SequenceInstance.getTagAlphabet();

		// List<SLFeatureVector> fvs = new ArrayList<>(instLength);
		SLFeatureVector fvs = new SLFeatureVector(-1, 0, null);

		// starting from the second token
		for (int i = 0; i < tags.length/* instLength */; i++) {
			int currTag = tagAlphabet.lookupIndex(tags[i]);
			int prevTag = (i == 0) ? -1 : tagAlphabet.lookupIndex(tags[i - 1]);

			// creating two features: (1) match of current and previous tags,
			// (2) match of tag and token index
			SLFeatureVector fv_ij = (i == 0) ? new SLFeatureVector(-1, -1.0, null)
					: sf.getFeatureVector(i, prevTag, currTag);
			SLFeatureVector fv_i = sf.getFeatureVector(i, currTag);
			// storing a concatenation of both
			fvs.addCat(fv_ij);
			fvs.addCat(fv_i);
			// fvs.add(SLFeatureVector.cat(fv_ij, fv_i));

		}
		// return getLabel(tags, fvs);
		return new SequenceLabel(tags, fvs);
	}

	// private SequenceLabel getLabel(String[] tags, List<SLFeatureVector> fvs)
	// {
	// return new SequenceLabel(tags, getFeatureVector(fvs, fvs.size() - 1));
	// }

	private SLFeatureVector getFeatureVector(List<SLFeatureVector> fvs, int idx) {
		if (idx == 0)
			return fvs.get(idx);

		return SLFeatureVector.cat(getFeatureVector(fvs, idx - 1), fvs.get(idx));
	}

}
