package struct.alg.clustering;

import static org.junit.Assert.assertTrue;

import java.util.Map;

import org.junit.Test;

import struct.sequence.SequenceLabel;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;
import weka.clusterers.HierarchicalClusterer;
import weka.core.Instance;

public class MAClustererTest {

	private final MAClusterer toTest = new MAClusterer(new HierarchicalClusterer());

	@Test
	public void testConvert() {
		SLLabel label = new SequenceLabel(new String[] { "protein", "cell_type" },
				new SLFeatureVector(1, 1.0, new SLFeatureVector(1, 2.0, new SLFeatureVector(2, 2.0, null))));
		Map<Integer, Double> map = toTest.extractFVs((SequenceLabel) label);
		map.remove(-1);

		Instance inst = toTest.convert(label, map);
		assertTrue(inst.numAttributes() == 2);
		assertTrue(inst.value(1) == 3);
	}
}
