package struct.alg.clustering;

import java.util.List;
import java.util.Map;

import struct.sequence.MACreator;
import struct.types.SLLabel;

/**
 * Merges
 *
 * @author ekravi
 *
 */
public interface MAMerger {
	Map<SLLabel, List<Integer>> merge(List<SLLabel> cluster, Map<SLLabel, List<Integer>> maMap, MACreator creator);
}
