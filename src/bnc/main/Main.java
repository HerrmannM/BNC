package bnc.main;

import java.security.SecureRandom;
import java.util.HashMap;

import org.apache.commons.math3.random.MersenneTwister;

import bnc.evaluator.Evaluator;
import bnc.evaluator.Holdout;
import bnc.evaluator.KFoldXVal;
import bnc.evaluator.TrainTest;
import bnc.evaluator.Evaluator.Result;
import bnc.model.ESKDB;
import bnc.model.Model;
import bnc.model.SKDB;


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
		
		TrainTest traintest = new TrainTest();
		evaluators.put(traintest.get_name(), traintest);
		
		
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
