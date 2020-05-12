package struct.alg;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import gurobi.GRB;
import org.junit.Before;
import org.junit.Test;

import gnu.trove.TIntDoubleHashMap;
import struct.alg.inference.GammaCalculator;
import struct.alg.inference.OptimizedGamma;
import struct.alg.inference.WMGamma;
import struct.sequence.SequenceInstance;
import struct.sequence.SequenceLabel;
import struct.types.Features;
import struct.types.Prediction;
import struct.types.SLFeatureVector;
import struct.types.SLInstance;
import struct.types.SLLabel;

public class RandomOptimizedGammaTest {

	private SLFeatureVector gold_fv;
	private SLFeatureVector pred_fv;
	private ArrayList<SLLabel> mJs;
	private double beta;
	private Predictor pred;
	private int dim = 20;
	@Before
	public void setUp() {
		createGoldAndPred();
		this.mJs = createMixedAssignments();
		createPredictor();
		this.beta = 1;
	}

	private void createPredictor() {

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
	private void createGoldAndPred() {
		this.gold_fv = new SLFeatureVector(-1, -1.0, null);
		this.pred_fv = new SLFeatureVector(-1, -1.0, null);
		Random intrand = new Random();
		// build feature vector with 3 features - (1,0,2) and (1,1,0)
		for(int i=0; i<this.dim; i++) {
			gold_fv.add(i,intrand.nextInt(10000));
			pred_fv.add(i,intrand.nextInt(10000));
		}
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
		TIntDoubleHashMap gold_hm = SLFeatureVector.Concentrate(gold_fv);
		TIntDoubleHashMap prev_hm = SLFeatureVector.Concentrate(pred_fv);
		Random boolrand = new Random();
		for(int i=0; i<this.dim; i++){
			SLFeatureVector mj_fv = new SLFeatureVector(-1, -1.0, null);
			for(int j=0; j<this.dim; j++){
				if(boolrand.nextBoolean())
					mj_fv.add(i, prev_hm.get(i));
				else
					mj_fv.add(i, gold_hm.get(i));
			}
			$.add(new SequenceLabel(null, mj_fv));
		}
		return $;
	}
	
	@Test
	public void randomTest(){
		GammaCalculator gamma_calc = new OptimizedGamma(gold_fv, pred_fv, mJs, pred, GRB.MINIMIZE);
		for (SLFeatureVector curr = gold_fv; curr != null; curr = curr.next)
			System.out.print(curr.value+" ");
		System.out.println();
		for (SLFeatureVector curr = pred_fv; curr != null; curr = curr.next)
			System.out.print(curr.value+" ");

		
	}
//
//	@Test
//	public void testSumWithNeutralBeta() {
//		WMGamma gc = new WMGamma(this.goldInst, this.mJs, 1.0, this.pred);
//		assertEquals(6.0, gc.sum, 0.0);
//	}
//
//	@Test
//	public void testSumWithSomeBeta() {
//		WMGamma gc = new WMGamma(this.goldInst, this.mJs, 2.452, this.pred);
//		assertEquals(35.4117, gc.sum, 0.005);
//	}
//
//	@Test
//	public void testGammaCalculation() {
//		WMGamma gc = new WMGamma(this.goldInst, this.mJs, 1.0, this.pred);
//		assertEquals(0.3333, gc.calculate(this.mJs.get(2)), 0.005);
//		assertEquals(0.6666, gc.calculate(this.mJs.get(1)), 0.005);
//		assertEquals(0.0, gc.calculate(this.mJs.get(0)), 0.0);
//	}
}
