package struct.alg.clustering;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import struct.sequence.MACreator;
import struct.sequence.SequenceLabel;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class MAClusterer implements MAMerger {

	private final AbstractClusterer clusterer;

	public MAClusterer(AbstractClusterer clusterer) {
		this.clusterer = clusterer;
	}

	public Map<SLLabel, List<Integer>> cluster(Map<SLLabel, List<Integer>> maMap, String[] options, MACreator creator)
			throws Exception {
		Instances data = toInstances(maMap.keySet());
		clusterer.setOptions(options);
		// clusterer.setDebug(true);
		clusterer.buildClusterer(data);

		return buildClusteredLabels(maMap, creator);
	}

	/**
	 * returns a list of SLLabels. Returned SLLabel is a union of all
	 * predictions
	 *
	 * @param creator
	 */
	private Map<SLLabel, List<Integer>> buildClusteredLabels(Map<SLLabel, List<Integer>> maMap, MACreator creator)
			throws Exception {
		Map<SLLabel, List<Integer>> $ = new HashMap<>();
		Map<Integer, List<SLLabel>> clusters = new HashMap<>();

		// retrieving clustered SLLabels
		for (SLLabel sl : maMap.keySet()) {
			Map<Integer, Double> map = extractFVs((SequenceLabel) sl);
			map.remove(-1);

			int cId = clusterer.clusterInstance(convert(sl, map));
			if (!clusters.containsKey(cId))
				clusters.put(cId, new LinkedList<>());
			clusters.get(cId).add(sl);
		}

		for (List<SLLabel> cluster : clusters.values()) {
			$.putAll(merge(cluster, maMap, creator));
		}
		return $;
	}

	/**
	 * Merges clustered SLLabels to a label with having prediction from all
	 * original labels
	 */

	@Override
	public Map<SLLabel, List<Integer>> merge(List<SLLabel> cluster, Map<SLLabel, List<Integer>> maMap,
			MACreator creator) {
		Map<SLLabel, List<Integer>> $ = new HashMap<>();

		String[] tags = ((SequenceLabel) cluster.get(0)).tags;
		List<Integer> indices = new ArrayList<>(cluster.size());
		for (SLLabel label : cluster) {
			List<Integer> idxList = maMap.get(label);
			for (Integer i : idxList) {
				tags[i] = ((SequenceLabel) label).tags[i];
			}
			indices.addAll(idxList);
		}

		SequenceLabel sl = creator.create(tags);
		$.put(sl, indices);

		return $;
	}

	// Convert to sparse instances including the non-zero feature vectors
	private Instances toInstances(Set<SLLabel> ma) {
		List<Instance> instList = new ArrayList<>(ma.size());
		int length = -1;
		for (SLLabel l : ma) {
			TreeMap<Integer, Double> map = extractFVs((SequenceLabel) l);
			map.remove(-1);
			length = Math.max(length, new ArrayList<>(map.keySet()).get(map.size() - 1));
			instList.add(convert(l, map));
		}

		length++;
		ArrayList<Attribute> attrs = new ArrayList<>(length);
		Instances $ = new Instances("genia", attrs, ma.size());
		for (int i = 0; i < length; ++i) {
			attrs.add(new Attribute(String.valueOf(i + 1)));
		}
		$.addAll(instList);

		return $;
	}

	Instance convert(SLLabel l, Map<Integer, Double> map) {
		int[] indices = new int[map.size()];
		double[] values = new double[map.size()];

		unbox(map, indices, values);
		Instance inst = new SparseInstance(1, values, indices, indices[indices.length - 1]);
		return inst;
	}

	private void unbox(Map<Integer, Double> map, int[] indices, double[] values) {
		int i = 0;
		for (Entry<Integer, Double> e : map.entrySet()) {
			indices[i] = e.getKey();
			values[i] = e.getValue();
			++i;
		}
	}

	TreeMap<Integer, Double> extractFVs(SequenceLabel sl) {
		TreeMap<Integer, Double> kv = new TreeMap<>();

		for (SLFeatureVector fv = sl.fv; fv != null; fv = fv.next) {
			int k = fv.index;
			double v = fv.value;

			if (!kv.containsKey(k))
				kv.put(k, 0.0);
			kv.put(k, kv.get(k) + v);
		}

		return kv;
	}
}
