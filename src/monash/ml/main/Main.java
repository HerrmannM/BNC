package monash.ml.main;

import java.security.SecureRandom;
import java.util.HashMap;

import org.apache.commons.math3.random.MersenneTwister;

import monash.ml.evaluator.Evaluator;
import monash.ml.evaluator.Holdout;
import monash.ml.evaluator.KFoldXVal;
import monash.ml.evaluator.Evaluator.Result;
import monash.ml.model.ESKDB;
import monash.ml.model.Model;
import monash.ml.model.SKDB;


public class Main {

	public static void main(String[] args) throws Exception {
		
		
		
		// --- --- --- List of available classifier
		HashMap<String, Model> models = new HashMap<>();
		
		SKDB skdb = new SKDB();
		models.put(skdb.getModelName(), skdb);
		
		ESKDB eskdb = new ESKDB();
		models.put(eskdb.getModelName(), eskdb);
		
		
		
		// --- --- --- List of available evaluators
		HashMap<String, Evaluator> evaluators = new HashMap<>();
		
		Holdout holdout = new Holdout();
		evaluators.put(holdout.get_name(), holdout);
		
		KFoldXVal kfoldxval = new KFoldXVal();
		evaluators.put(kfoldxval.get_name(), kfoldxval);

		
		
		// --- --- --- Read the command line, handle options.. Exit on error.
		CLIParsing opt = new CLIParsing(args, models, evaluators);
		Model learner = opt.getModel();
		Evaluator eval = opt.getEvaluator();
	
		
		
		// --- --- --- Evaluate the model and print the result
		// --- Random and secure random
		long seed = 3071980;
		MersenneTwister rg = new MersenneTwister(seed);
		SecureRandom sr = SecureRandom.getInstanceStrong();
		
		// --- Launch
		Result res = eval.evaluate(learner, rg, sr, true);
		System.out.println(res.toString());
	}

}
