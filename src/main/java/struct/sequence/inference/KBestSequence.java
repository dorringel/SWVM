package struct.sequence.inference;

import cc.mallet.types.Alphabet;
import struct.sequence.SequenceLabel;
import struct.sequence.SequencePrediction;
import struct.types.SLFeatureVector;

/**
 * KBestSequence.
 */
public class KBestSequence {

	// chart - stores a feature vector for each triplet of number of tokens
	// (terms), number of tags, K
	private final SequenceItem[][][] chart;
	private final int K;
	private final int num_tags;
	private final Alphabet tagAlphabet;

	public KBestSequence(Alphabet tagAlphabet, int instLength, int K) {
		this.K = K;
		chart = new SequenceItem[instLength][tagAlphabet.size()][K];
		num_tags = tagAlphabet.size();
		this.tagAlphabet = tagAlphabet;
	}

	/**
	 * adds a new feature to the chart
	 *
	 * @return the index of the feature in the chart or -1 if it didn't got it
	 */
	public int add(int n, int e, double prob, SLFeatureVector fv, SequenceItem prev, int strt) {

		if (chart[n][e][0] == null) {
			for (int i = 0; i < K; i++)
				chart[n][e][i] = new SequenceItem(n, e, Double.NEGATIVE_INFINITY, null, null);
		}

		// early pruning - probability lower than last item skip this
		// addition
		if (chart[n][e][K - 1].prob > prob)
			return -1;

		// pushing new feature to the best position (up to k) according to
		// its probability
		for (int i = strt; i < K; i++) {
			if (chart[n][e][i].prob < prob) {
				SequenceItem tmp = chart[n][e][i];
				chart[n][e][i] = new SequenceItem(n, e, prob, fv, prev);
				for (int j = i + 1; j < K && tmp.prob != Double.NEGATIVE_INFINITY; j++) {
					SequenceItem tmp1 = chart[n][e][j];
					chart[n][e][j] = tmp;
					tmp = tmp1;
				}
				return i + 1 >= K ? -1 : i + 1;
			}
		}
		return -1;
	}

	// private double getProb(int n, int e) {
	// return getProb(n, e, 0);
	// }

	// private double getProb(int n, int e, int i) {
	// if (chart[n][e][i] != null)
	// return chart[n][e][i].prob;
	// return Double.NEGATIVE_INFINITY;
	// }

	// private double[] getProbs(int n, int e) {
	// double[] result = new double[K];
	// for (int i = 0; i < K; i++)
	// result[i] = chart[n][e][i] != null ? chart[n][e][i].prob :
	// Double.NEGATIVE_INFINITY;
	// return result;
	// }
	//
	// private SequenceItem getItem(int n, int e) {
	// return getItem(n, e, 0);
	// }

	// private SequenceItem getItem(int n, int e, int i) {
	// if (chart[n][e][i] != null)
	// return chart[n][e][i];
	// return null;
	// }

	public SequenceItem[] getItems(int n, int e) {
		if (chart[n][e][0] != null)
			return chart[n][e];
		return null;
	}

	public SequencePrediction getBestSequences() {

		int n = chart.length - 1;

		SequenceItem[] best = new SequenceItem[K];
		for (int i = 0; i < K; i++) {
			best[i] = new SequenceItem(-1, -1, Double.NEGATIVE_INFINITY, null, null);
		}

		for (int e = 0; e < num_tags; e++) {
			for (int k = 0; k < K; k++) {
				SequenceItem cand = chart[n][e][k];

				for (int i = 0; i < K; i++) {
					if (best[i].prob < cand.prob) {
						SequenceItem tmp = best[i];
						best[i] = cand;
						for (int j = i + 1; j < K && tmp.prob != Double.NEGATIVE_INFINITY; j++) {
							SequenceItem tmp1 = best[j];
							best[j] = tmp;
							tmp = tmp1;
						}
						break;
					}
				}

			}
		}

		SequenceLabel[] d = new SequenceLabel[K];
		for (int k = 0; k < K; k++) {
			if (best[k].prob != Double.NEGATIVE_INFINITY) {
				d[k] = new SequenceLabel(getEntString(best[k]).split(" "), getFeatureVector(best[k]));
			} else {
				d[k] = null;
			}
		}
		return new SequencePrediction(d);
	}

	private SLFeatureVector getFeatureVector(SequenceItem si) {
		if (si.prev == null)
			return si.fv;

		return SLFeatureVector.cat(getFeatureVector(si.prev), si.fv);
	}

	private String getEntString(SequenceItem si) {
		if (si.prev == null)
			return ((String) tagAlphabet.lookupObject(si.e)).trim();

		return (getEntString(si.prev) + " " + (String) tagAlphabet.lookupObject(si.e)).trim();
	}
}
