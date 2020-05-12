package struct.alg.inference;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import gnu.trove.TIntDoubleHashMap;
import struct.alg.Predictor;
import struct.sequence.SequenceInstance;
import struct.sequence.SequenceLabel;
import struct.types.SLFeatureVector;
import struct.types.SLLabel;

import gurobi.*;

public class OptimizedGamma implements GammaCalculator {
	double[] gammaArray;
	HashMap<SLLabel, Double> maToGammaMap;
	static GRBEnv env = null;
	boolean didSolve = true;

	private SLFeatureVector gold_fv;
	private SLFeatureVector pred_fv;
	private List<SLLabel> JJs;
	private Predictor pred;
	private int gammaObjectiveType;

	/**
	 *
	 * @param gold_fv - true output [sequence linear feature vector]
	 * @param pred_fv - predicted output [sequence linear feature vector]
	 * @param JJs - mixed-assignments [linked list]
	 * @param pred - predictor (which contains W) [sequence predictor]
	 * @param gammaObjectiveType - "MINIMIZE" / "MAXIMIZE"
	 */
	public OptimizedGamma(SLFeatureVector gold_fv, SLFeatureVector pred_fv ,List<SLLabel> JJs, Predictor pred, int gammaObjectiveType) {
		//try {
			this.gold_fv = gold_fv;
			this.pred_fv = pred_fv;
			this.JJs = JJs;
			this.pred = pred;
			this.gammaObjectiveType = gammaObjectiveType;
		//} catch (Exception e) {
//		didSolve = false;
//		System.out.println("Error message: " + e.getMessage() + ". " +
//				e.getMessage());
//		}
	}

	/**
	 * Standard optimization formulation
	 * (number #1 in Rotem's formulations page)
	 */
	public HashMap<SLLabel, Double> optimizationOne(){

		try {
			if(env == null) {
				// Create a new environment
				System.out.println("Creating a new environment");
				env = new GRBEnv();

				//Turning off the output - set this flag to be 1 (or just remove the env.set) to see the model outputs
				env.set(GRB.IntParam.OutputFlag, 0);
			}

			// Define a model and name it
			GRBModel model = new GRBModel(env);
			model.set(GRB.StringAttr.ModelName, "gamma");

			// Number of assignments is same to the number of Mixed Assignments
			int nAssignments = JJs.size();

			// Create the variables for the gamma for each assignment.
			GRBVar[] gammas = new GRBVar[nAssignments];
			for (int i=0; i<nAssignments; i++) {
				// gammas[i] = model.addVar(lowerBound, upperBound, objectiveCoefficient, type, name);
				// TODO: investigate whether objectiveCoefficient should indeed be zero
				gammas[i] = model.addVar(0.0, 1.0, 1, GRB.CONTINUOUS, "gamma_"+i);
			}
			
			// Integrate the new variables
			model.update();

			//Create a hash-map for each assignment, and collect all the keys in every mixed assignment distance vector.
			//We will use these hash-maps for efficient constraint and objective creation.
			//We assume that all the features in the indices that are not in those keys are 0.
			TIntDoubleHashMap[] hms = new TIntDoubleHashMap[nAssignments];
			Set<Integer> keySet = new HashSet<>();
			for(int i=0; i<nAssignments; i++){

				// get MA's feature vector
				SLFeatureVector ma_fv = JJs.get(i).getFeatureVectorRepresentation();

				// Calculate distance gold_fv - ma_fv
				SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, ma_fv);

				// Add the hashmap of the distance vector
				hms[i] = SLFeatureVector.Concentrate(dist_fv);

				// Arrays.asList(hms[i].keys());

				// Add the keys of the hashmap to the key collection.
				for(int j=0; j<hms[i].keys().length; j++)
					keySet.add(Integer.valueOf(hms[i].keys()[j]));
			}


			// Set the objective function
			GRBLinExpr objective_expr = new GRBLinExpr();
			TIntDoubleHashMap pred_gold_dist_hm = SLFeatureVector.Concentrate(SLFeatureVector.getDistVector(gold_fv, pred_fv));
			for(int i=0; i<pred_gold_dist_hm.keys().length; i++) {
				//In this expr we will put w_old[i] + sum(gamma_j*phi_j[i])[i]
				GRBLinExpr weight_expr = new GRBLinExpr();
				double[] deltaVals = new double[nAssignments];
				//for every key i, calculate an array of delta_phi[i].
				for(int j=0; j<nAssignments; j++)
					deltaVals[j] = (hms[j].containsKey(i) ? hms[j].get(i) : 0);
				weight_expr.addTerms(deltaVals, gammas);
				weight_expr.addConstant(pred.weights[i]);
				//weight_expr the new post update w[i], so we add w[i]*delta_gold_pred[i]
				objective_expr.multAdd(pred_gold_dist_hm.get(pred_gold_dist_hm.keys()[i]), weight_expr);
			}
			

			// Determine objective type by parameter gammaObjectiveType ["MINIMIZE" / "MAXIMIZE"]
			model.setObjective(objective_expr, gammaObjectiveType);

			// Selection condition (1)

			/// Add constraint: sum(gammas)=1
//			GRBLinExpr expr = new GRBLinExpr();
//			double[] coeffs = new double[nAssignments];
//			Arrays.fill(coeffs, 1);
//			expr.addTerms(coeffs, gammas);
//			model.addConstr(expr, GRB.EQUAL, 1, "gamma_sum_equals_one");

			/// Add constraint: gamma[i]>=0 for all i
//			for (int j = 0; j <  nAssignments; j++) {
//				GRBLinExpr nonnegativity_expr = new GRBLinExpr();
//				nonnegativity_expr.addTerm(1, gammas[j]);
//				model.addConstr(nonnegativity_expr, GRB.GREATER_EQUAL, 0, "gamma_non_negative"+j);
//			}

			// Selection condition (2)

			/// Add constraints - w_old * sum(gamma_j*(delta phi(x,y_gold,mj)) <= 0

			/// An expression to contain multiple weight*sum expressions
			GRBLinExpr master_expr = new GRBLinExpr();

			/// Iterate over whatever keys we have
			for(Integer i : keySet) {
				GRBLinExpr weight_expr = new GRBLinExpr();
				double[] deltaVals = new double[nAssignments];
				//for every key i, calculate an array of delta_phi[i].
				for(int j=0; j<nAssignments; j++)
					deltaVals[j] = (hms[j].containsKey(i.intValue()) ? hms[j].get(i.intValue()) : 0);
				weight_expr.addTerms(deltaVals, gammas);
				//add weights[i] * sum(delta_phi_j[i]*gamma_j)
				master_expr.multAdd(pred.weights[i], weight_expr);
			}
			model.addConstr(master_expr, GRB.LESS_EQUAL, 0, "total_is_violation");
					
			
			// Perform optimization
			model.optimize();

			gammaArray = new double[nAssignments];
			maToGammaMap = new HashMap<>();
			for (int i=0; i<nAssignments; i++) {
				GRBVar g = gammas[i];
				double currGamma = g.get(GRB.DoubleAttr.X);
				gammaArray[i] = currGamma;
				maToGammaMap.put(JJs.get(i), currGamma);

//				System.out.println(g.get(GRB.StringAttr.VarName) + " " + g.get(GRB.DoubleAttr.X));
			}
//			System.out.println("Obj: " + model.get(GRB.DoubleAttr.ObjVal));
			model.dispose();
//			env.dispose();

			return maToGammaMap;

	    } catch (NullPointerException e) {
			System.out.println("## ERROR: NullPointerException in OptimizedGamma");
			System.out.println(e.getCause().toString());
			System.exit(1);
		} catch (GRBException e) {
	    	didSolve = false;
	        System.out.println("Error code: " + e.getErrorCode() + ". " +
	            e.getMessage());
		}

		return maToGammaMap;
	}

	/**
	 * A variant of the original optimization formulation
	 * switch between m_j and y_pred
	 * (number #4 in Rotem's formulations page)
	 */
	public HashMap<SLLabel, Double> optimizationFour(){

		try {
			if(env == null) {
				// Create a new environment
				System.out.println("Creating a new environment");
				env = new GRBEnv();

				//Turning off the output - set this flag to be 1 (or just remove the env.set) to see the model outputs
				env.set(GRB.IntParam.OutputFlag, 0);
			}

			// Define a model and name it
			GRBModel model = new GRBModel(env);
			model.set(GRB.StringAttr.ModelName, "gamma");

			// Number of assignments is same to the number of Mixed Assignments
			int nAssignments = JJs.size();

			// Create the variables for the gamma for each assignment.
			GRBVar[] gammas = new GRBVar[nAssignments];
			for (int i=0; i<nAssignments; i++) {
				// gammas[i] = model.addVar(lowerBound, upperBound, objectiveCoefficient, type, name);
				// TODO: investigate whether objectiveCoefficient should indeed be zero
				gammas[i] = model.addVar(0.0, 1.0, 1, GRB.CONTINUOUS, "gamma_"+i);
			}

			// Integrate the new variables
			model.update();

			//Create a hash-map for each assignment, and collect all the keys in every mixed assignment distance vector.
			//We will use these hash-maps for efficient constraint and objective creation.
			//We assume that all the features in the indices that are not in those keys are 0.
			TIntDoubleHashMap[] hms = new TIntDoubleHashMap[nAssignments];
			Set<Integer> keySet = new HashSet<>();
			for(int i=0; i<nAssignments; i++){

				// get MA's feature vector
				SLFeatureVector ma_fv = JJs.get(i).getFeatureVectorRepresentation();

				// Calculate distance gold_fv - ma_fv
				SLFeatureVector dist_fv = SLFeatureVector.getDistVector(gold_fv, ma_fv);

				// Add the hashmap of the distance vector
				hms[i] = SLFeatureVector.Concentrate(dist_fv);

				// Arrays.asList(hms[i].keys());

				// Add the keys of the hashmap to the key collection.
				for(int j=0; j<hms[i].keys().length; j++)
					keySet.add(Integer.valueOf(hms[i].keys()[j]));
			}

			/// An expression to contain multiple weight*sum expressions
			GRBLinExpr objective_expr = new GRBLinExpr();

			/// Iterate over whatever keys we have
			for(Integer i : keySet) {
				GRBLinExpr inner_expr = new GRBLinExpr();
				double[] deltaVals = new double[nAssignments];
				//for every key i, calculate an array of delta_phi[i].
				for(int j=0; j<nAssignments; j++)
					deltaVals[j] = (hms[j].containsKey(i.intValue()) ? hms[j].get(i.intValue()) : 0);
				inner_expr.addTerms(deltaVals, gammas);
				//add weights[i] * sum(delta_phi_j[i]*gamma_j)
				objective_expr.multAdd(pred.weights[i], inner_expr);
			}

			// optimizationOne() objective function
			/*
			GRBLinExpr objective_expr = new GRBLinExpr();
			TIntDoubleHashMap pred_gold_dist_hm = SLFeatureVector.Concentrate(SLFeatureVector.getDistVector(gold_fv, pred_fv));
			for(int i=0; i<pred_gold_dist_hm.keys().length; i++) {
				//In this expr we will put w_old[i] + sum(gamma_j*phi_j[i])[i]
				GRBLinExpr weight_expr = new GRBLinExpr();
				double[] deltaVals = new double[nAssignments];
				//for every key i, calculate an array of delta_phi[i].
				for(int j=0; j<nAssignments; j++)
					deltaVals[j] = (hms[j].containsKey(i) ? hms[j].get(i) : 0);
				weight_expr.addTerms(deltaVals, gammas);
				weight_expr.addConstant(pred.weights[i]);
				//weight_expr the new post update w[i], so we add w[i]*delta_gold_pred[i]
				objective_expr.multAdd(pred_gold_dist_hm.get(pred_gold_dist_hm.keys()[i]), weight_expr);
			}
			*/
			// Set the objective function
			/// Add constraints - w_old * sum(gamma_j*(delta phi(x,y_gold,mj)) <= 0


			// Determine objective type by parameter gammaObjectiveType ["MINIMIZE" / "MAXIMIZE"]
			model.setObjective(objective_expr, gammaObjectiveType);

			// Selection condition (1)

			/// Add constraint: sum(gammas)=1
			GRBLinExpr expr = new GRBLinExpr();
			double[] coeffs = new double[nAssignments];
			Arrays.fill(coeffs, 1);
			expr.addTerms(coeffs, gammas);
			model.addConstr(expr, GRB.EQUAL, 1, "gamma_sum_equals_one");

			/// Add constraint: gamma[i]>=0 for all i
//			for (int j = 0; j <  nAssignments; j++) {
//				GRBLinExpr nonnegativity_expr = new GRBLinExpr();
//				nonnegativity_expr.addTerm(1, gammas[j]);
//				model.addConstr(nonnegativity_expr, GRB.GREATER_EQUAL, 0, "gamma_non_negative"+j);
//			}

			// Selection condition (2)

			/// Add constraints - w_old * sum(gamma_j*(delta phi(x,y_gold,mj)) <= 0

			/// An expression to contain multiple weight*sum expressions
			GRBLinExpr master_expr = new GRBLinExpr();

			/// Iterate over whatever keys we have
			for(Integer i : keySet) {
				GRBLinExpr weight_expr = new GRBLinExpr();
				double[] deltaVals = new double[nAssignments];
				//for every key i, calculate an array of delta_phi[i].
				for(int j=0; j<nAssignments; j++)
					deltaVals[j] = (hms[j].containsKey(i.intValue()) ? hms[j].get(i.intValue()) : 0);
				weight_expr.addTerms(deltaVals, gammas);
				//add weights[i] * sum(delta_phi_j[i]*gamma_j)
				master_expr.multAdd(pred.weights[i], weight_expr);
			}
			model.addConstr(master_expr, GRB.LESS_EQUAL, 0, "total_is_violation");


			// Perform optimization
			model.optimize();

			gammaArray = new double[nAssignments];
			maToGammaMap = new HashMap<>();
			for (int i=0; i<nAssignments; i++) {
				GRBVar g = gammas[i];
				double currGamma = g.get(GRB.DoubleAttr.X);
				gammaArray[i] = currGamma;
				maToGammaMap.put(JJs.get(i), currGamma);

//				System.out.println(g.get(GRB.StringAttr.VarName) + " " + g.get(GRB.DoubleAttr.X));
			}
//			System.out.println("Obj: " + model.get(GRB.DoubleAttr.ObjVal));
			model.dispose();
//			env.dispose();

			return maToGammaMap;

		} catch (NullPointerException e) {
			System.out.println("## ERROR: NullPointerException in OptimizedGamma");
			System.out.println(e.getCause().toString());
			System.exit(1);
		} catch (GRBException e) {
			didSolve = false;
			System.out.println("Error code: " + e.getErrorCode() + ". " +
					e.getMessage());
		}

		return maToGammaMap;

	}


	@Override
	public double calculate(SLLabel mJ) {
		//If we couldn't solve the problem for some reason, gamma=0 for everything.
		if(!didSolve)
			return 0;
		return maToGammaMap.get(mJ);
	}
	

	private static void printarr(double[] arr) {
		for(int i=0; i<arr.length-1; i++)
			System.out.print(arr[i]+", ");
		System.out.println(arr[arr.length-1]);
	}
	
	/**
	 * A utility function to debug the optimizer. this prints out the weights, 
	 * the features of the golden_fv, pred_fv and then the fv for every MA.
	 * @param goldInst
	 * @param gold_fv
	 * @param pred_fv
	 * @param JJs
	 * @param pred
	 */
	public static void printInputToOptimizer(SequenceInstance goldInst, SLFeatureVector gold_fv, SLFeatureVector pred_fv ,List<SLLabel> JJs, Predictor pred){
		int highest_feature = 0;
		TIntDoubleHashMap ghm = SLFeatureVector.Concentrate(gold_fv);
		int[] gkeys = ghm.keys();
		for(int i=0; i<gkeys.length; i++){
			if(gkeys[i] > highest_feature)
				highest_feature = gkeys[i];
		}
		
		TIntDoubleHashMap phm = SLFeatureVector.Concentrate(pred_fv);
		int[] pkeys = phm.keys();
		for(int i=0; i<pkeys.length; i++){
			if(pkeys[i] > highest_feature){
				highest_feature = pkeys[i];
			}
		}
		
		
		double[] gvals = new double[highest_feature+1];
		double[] pvals = new double[highest_feature+1];
		
		for(int i=0; i<gvals.length; i++) {
			if(ghm.containsKey(i))
				gvals[i] = ghm.get(i);
			else
				gvals[i] = 0;
			
			if(phm.containsKey(i))
				pvals[i] = phm.get(i);
			else
				pvals[i] = 0;
		}
		
		for(int i=0; i<highest_feature; i++){
			System.out.print(pred.weights[i]+", ");
		}
		System.out.println(pred.weights[highest_feature+1]);
		printarr(gvals);
		printarr(pvals);
		for(int i=0; i<goldInst.getLabel().tags.length; i++){
			System.out.print(goldInst.getLabel().tags[i]+" ");
		}
		System.out.println("\n-----------------------------------");
		
		for(SLLabel mj : JJs){
			TIntDoubleHashMap mjhm = SLFeatureVector.Concentrate(mj.getFeatureVectorRepresentation());
			double[] mjvals = new double[highest_feature+1];
			int[] mjkeys = mjhm.keys();
			for (int i=0; i<mjvals.length; i++){
				if(mjhm.containsKey(i))
					mjvals[i] = mjhm.get(i);
				else
					mjvals[i] = 0;
			}
			SequenceLabel slmj =  (SequenceLabel)mj;
			printarr(mjvals);
		}
		for(SLLabel mj : JJs){
			TIntDoubleHashMap mjhm = SLFeatureVector.Concentrate(mj.getFeatureVectorRepresentation());
			double[] mjvals = new double[highest_feature+1];
			int[] mjkeys = mjhm.keys();
			for (int i=0; i<mjvals.length; i++){
				if(mjhm.containsKey(i))
					mjvals[i] = mjhm.get(i);
				else
					mjvals[i] = 0;
			}
			SequenceLabel slmj =  (SequenceLabel)mj;
			for (int i=0; i< slmj.tags.length; i++)
				System.out.print (slmj.tags[i]+" ");
			System.out.println();

		}

		
	}

	public double[] getGammaArray(){
		return this.gammaArray;
	}

}
