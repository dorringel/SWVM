package struct.alg;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import struct.alg.inference.WMGamma;
import struct.sequence.SequenceInstance;
import struct.sequence.SequenceLabel;
import struct.types.Features;
import struct.types.Prediction;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;
import struct.types.SLLabel;

public class AggressiveWMGammaTest {

	private SequenceInstance goldInst;
	private ArrayList<SLLabel> mJs;
	private double beta;
	private Predictor pred;

	@Before
	public void setUp() {
		createGoldInstance();
		this.mJs = createMixedAssignments();
		createPredictor();
		this.beta = 1;
	}

	private void createPredictor() {
		int dim = 3;
		this.pred = new Predictor(dim) {

			@Override
			public void grow(int newSize) {
				// TODO Auto-generated method stub

			}

			@Override
			public Prediction decode(SLInstance inst, Features feats, int K) {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public Prediction decode(SLInstance inst, Features feats) {
				// TODO Auto-generated method stub
				return null;
			}
		};

		// Setting 2 as the weight (arbitrary)
		for (int i = 0; i < dim; i++) {
			this.pred.weights[i] = 2.0;
			this.pred.avg_weights[i] = 2.0;
		}
	}

	/**
	 * Create a gold instance feature vector - (1,0,2)
	 */
	private void createGoldInstance() {
		SLFeatureVector fv1 = new SLFeatureVector(-1, -1.0, null);
		// build feature vector with 3 features - (1,0,2) and (1,1,0)
		fv1.add(0, 1);
		fv1.add(1, 0);
		fv1.add(2, 2);
		this.goldInst = new SequenceInstance(null, null, fv1, null);
	}

	/**
	 * Create mixed assignments for the gold instance. per index, create a new
	 * mixed assignment, swapping the value in the index with 2 (assuming 2 is
	 * the prediction value)
	 *
	 * @return
	 */
	private ArrayList<SLLabel> createMixedAssignments() {
		ArrayList<SLLabel> $ = new ArrayList<SLLabel>();
		for (SLFeatureVector gold = this.goldInst.getLabel()
				.getFeatureVectorRepresentation(); gold != null; gold = gold.next) {
			if (gold.index < 0) {
				continue;
			}
			SLFeatureVector mj_fv = new SLFeatureVector(-1, -1.0, null);
			for (SLFeatureVector curr = this.goldInst.getLabel()
					.getFeatureVectorRepresentation(); curr != null; curr = curr.next) {
				if (gold.index < 0) {
					continue;
				}
				if (curr.index == gold.index && curr.value == gold.value) {
					// We're swapping for a constant value
					mj_fv.add(curr.index, 2);
				} else {
					mj_fv.add(curr.index, curr.value);
				}
			}
			$.add(new SequenceLabel(null, mj_fv));
		}
		return $;
	}

	@Test
	public void testSumWithNeutralBeta() {
		WMGamma gc = new WMGamma(this.goldInst, this.mJs, 1.0, this.pred);
		assertEquals(6.0, gc.sum, 0.0);
	}

	@Test
	public void testSumWithSomeBeta() {
		WMGamma gc = new WMGamma(this.goldInst, this.mJs, 2.452, this.pred);
		assertEquals(35.4117, gc.sum, 0.005);
	}

	@Test
	public void testGammaCalculation() {
		WMGamma gc = new WMGamma(this.goldInst, this.mJs, 1.0, this.pred);
		assertEquals(0.3333, gc.calculate(this.mJs.get(2)), 0.005);
		assertEquals(0.6666, gc.calculate(this.mJs.get(1)), 0.005);
		assertEquals(0.0, gc.calculate(this.mJs.get(0)), 0.0);
	}
}
