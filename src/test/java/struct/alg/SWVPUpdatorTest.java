package struct.alg;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import struct.sequence.SequenceInstance;
import struct.sequence.SequenceLabel;
import struct.sequence.SequencePrediction;
import struct.types.SLLabel;

public class SWVPUpdatorTest {

	private final SWVPUpdator toTest = new SWVPUpdator();

	// @Test
	public void testUpdate() {
		fail("Not yet implemented");
	}

	@Test
	public void testGetMixedAssignmentPerIndex() {
		SequenceInstance goldInst = new SequenceInstance(null, new String[] { "O", "O" }, null, null);
		SequencePrediction sqPred = new SequencePrediction(
				new SequenceLabel[] { new SequenceLabel(new String[] { "protein", "cell_type" }, null) });
		Map<SLLabel, List<Integer>> mapi = toTest.getSingleMixedAssignments(goldInst, sqPred);

		assertTrue(mapi.size() == goldInst.getLabel().tags.length);
		for (int i = 0; i < goldInst.getLabel().tags.length; ++i) {
			SequenceLabel sl = (SequenceLabel) mapi.get(i);
			assertTrue(sl.tags[i].equals(sqPred.getBestLabel().tags[i]));
		}
	}

	@Test
	public void testGetJJ() {
		// sequence prediction in size of 1
		String[] jj = toTest.getMA(new String[] { "O", "O" }, new String[] { "protein", "cell_type" }, 0);

		assertTrue(Arrays.equals(new String[] { "protein", "O" }, jj));
	}
}
