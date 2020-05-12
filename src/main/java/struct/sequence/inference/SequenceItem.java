package struct.sequence.inference;

import struct.types.SLFeatureVector;

/**
 * Item for a sequence.
 */
public class SequenceItem {

	public int n, e;
	public double prob;
	public SLFeatureVector fv;
	public SequenceItem prev;

	public SequenceItem(int n, int e, double prob, SLFeatureVector fv, SequenceItem prev) {

		this.n = n;
		this.e = e;
		this.prob = prob;
		this.fv = fv;
		this.prev = prev;
	}
}