/* Copyright (C) 2006 University of Pennsylvania.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

package struct.types;

import java.util.ArrayList;
import java.util.Arrays;

import cc.mallet.types.Alphabet;
import gnu.trove.TIntDoubleHashMap;

/**
 * The feature vector.
 *
 * @version 08/15/2006
 */
public class SLFeatureVector implements Comparable {

	public int index;
	public double value;
	public SLFeatureVector next;

	public SLFeatureVector(int i, double v, SLFeatureVector n) {
		index = i;
		value = v;
		next = n;
	}

	/**
	 * Adds itself to another SLFeatureVector if the other feature is valid.
	 */
	public SLFeatureVector add(String feat, double val, Alphabet dataAlphabet) {
		int num = dataAlphabet.lookupIndex(feat);
		if (num >= 0) {
			return new SLFeatureVector(num, val, this);
		}
		return this;
	}

	/**
	 * Takes the new parameters and becomes the head of the chain.
	 */
	public void add(int i1, double v1) {
		SLFeatureVector new_node = new SLFeatureVector(this.index, this.value, this.next);

		this.index = i1;
		this.value = v1;
		this.next = new_node;
	}
	

	public static int getMaxIndex(SLFeatureVector fv){
		int maxIndex = -1;
		for (SLFeatureVector curr = fv; curr.next != null; curr = curr.next) {
			if (curr.index > maxIndex)
				maxIndex = curr.index;
		}
		return maxIndex;

	}
	
	/**
	 * I hereby declare this method - a non retarded concat that doesn't work in o(n^shitload) when trying to concat many vectors.
	 */
	public void addCat(SLFeatureVector fv) {
		for (SLFeatureVector curr = fv; curr.next != null; curr = curr.next) {
			if(curr.index < 0){
				continue;
			}
			add(curr.index, curr.value);
		}
	}
	

	/**
	 * Concatenates both the SLFeatureVectors
	 */
	public static SLFeatureVector cat(SLFeatureVector fv1, SLFeatureVector fv2) {
		SLFeatureVector result = new SLFeatureVector(-1, -1.0, null);
		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			result = new SLFeatureVector(curr.index, curr.value, result);
		}
		for (SLFeatureVector curr = fv2; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			result = new SLFeatureVector(curr.index, curr.value, result);
		}
		return result;

	}

	/**
	 * Creates and returns the distance vector(fv1 - fv2) of both the
	 * SLFeatureVectors.
	 */
	public static SLFeatureVector getDistVector(SLFeatureVector fv1, SLFeatureVector fv2) {
		SLFeatureVector result = new SLFeatureVector(-1, -1.0, null);
		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			result = new SLFeatureVector(curr.index, curr.value, result);
		}
		for (SLFeatureVector curr = fv2; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			result = new SLFeatureVector(curr.index, -curr.value, result);
		}
		return result;
	}

	/**
	 * Computes the dot product of both the SLFeatureVectors.
	 */
	public static double dotProduct(SLFeatureVector fv1, SLFeatureVector fv2) {
		double result = 0.0;
		TIntDoubleHashMap hm1 = new TIntDoubleHashMap();
		TIntDoubleHashMap hm2 = new TIntDoubleHashMap();

		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			hm1.put(curr.index, hm1.get(curr.index) + curr.value);
		}
		for (SLFeatureVector curr = fv2; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			hm2.put(curr.index, hm2.get(curr.index) + curr.value);
		}

		int[] keys = hm1.keys();

		for (int i = 0; i < keys.length; i++) {
			double v1 = hm1.get(keys[i]);
			double v2 = hm2.get(keys[i]);
			result += v1 * v2;
		}
		return result;
	}

	/**
	 * Computes the first norm of the SLFeatureVector.
	 */
	public static double oneNorm(SLFeatureVector fv1) {
		double sum = 0.0;
		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			sum += curr.value;
		}
		return sum;
	}

	/**
	 * Computes the size of the SLFeatureVector.
	 */
	public static int size(SLFeatureVector fv1) {
		int sum = 0;
		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			sum++;
		}
		return sum;
	}

	/**
	 * Computes the second norm of the SLFeatureVector.
	 */
	public static double twoNorm(SLFeatureVector fv1) {
		TIntDoubleHashMap hm = new TIntDoubleHashMap();
		double sum = 0.0;
		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			hm.put(curr.index, hm.get(curr.index) + curr.value);
		}
		int[] keys = hm.keys();

		for (int i = 0; i < keys.length; i++)
			sum += Math.pow(hm.get(keys[i]), 2.0);

		return Math.sqrt(sum);
	}

	/**
	 * Normalizes the SLFeatureVector with its second norm.
	 */
	public static SLFeatureVector twoNormalize(SLFeatureVector fv1) {
		return normalize(fv1, twoNorm(fv1));
	}

	/**
	 * Normalizes the SLFeatureVector with its first norm.
	 */
	public static SLFeatureVector oneNormalize(SLFeatureVector fv1) {
		return normalize(fv1, oneNorm(fv1));
	}

	/**
	 * Normalizes the SLFeatureVector with the given norm.
	 */
	public static SLFeatureVector normalize(SLFeatureVector fv1, double norm) {
		SLFeatureVector result = new SLFeatureVector(-1, -1.0, null);
		for (SLFeatureVector curr = fv1; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			result = new SLFeatureVector(curr.index, curr.value / norm, result);
		}
		return result;
	}

	@Override
	public String toString() {
		if (next == null)
			return "" + index + ":" + value;
		return index + ":" + value + " " + next.toString();
	}

	public void sort() {
		ArrayList features = new ArrayList();

		for (SLFeatureVector curr = this; curr != null; curr = curr.next)
			if (curr.index >= 0)
				features.add(curr);

		Object[] feats = features.toArray();

		Arrays.sort(feats);

		SLFeatureVector fv = new SLFeatureVector(-1, -1.0, null);
		for (int i = feats.length - 1; i >= 0; i--) {
			SLFeatureVector tmp = (SLFeatureVector) feats[i];
			fv = new SLFeatureVector(tmp.index, tmp.value, fv);
		}

		this.index = fv.index;
		this.value = fv.value;
		this.next = fv.next;

	}

	public int compareTo(Object o) {
		SLFeatureVector fv = (SLFeatureVector) o;
		if (index < fv.index)
			return -1;
		if (index > fv.index)
			return 1;
		return 0;
	}

	/**
	 * Computes the dot product of this SLFeatureVector with the given weights.
	 */
	public double dotProdoct(double[] weights) {
		double score = 0.0;
		for (SLFeatureVector curr = this; curr != null; curr = curr.next) {
			if (curr.index >= 0)
				score += weights[curr.index] * curr.value;
		}
		return score;
	}
	

	/**
	 * Computes the dot product of both the SLFeatureVectors.
	 */
	public static int CountNonZeros(SLFeatureVector fv) {
		int nNonZeros = 0;
		TIntDoubleHashMap hm = new TIntDoubleHashMap();

		for (SLFeatureVector curr = fv; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			hm.put(curr.index, hm.get(curr.index) + curr.value);
		}

		int[] keys = hm.keys();

		for (int i = 0; i < keys.length; i++) {
			if(hm.get(keys[i]) != 0)
				nNonZeros++;
		}
		return nNonZeros;
	}
	
	/**
	 * Concentrate a feature vector into a hash map.
	 */
	public static TIntDoubleHashMap Concentrate(SLFeatureVector fv) {
		int nNonZeros = 0;
		TIntDoubleHashMap hm = new TIntDoubleHashMap();

		for (SLFeatureVector curr = fv; curr.next != null; curr = curr.next) {
			if (curr.index < 0)
				continue;
			hm.put(curr.index, hm.get(curr.index) + curr.value);
		}
		return hm;
	}
}
