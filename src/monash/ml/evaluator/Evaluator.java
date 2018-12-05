package monash.ml.evaluator;

import java.security.SecureRandom;

import org.apache.commons.cli.*;
import org.apache.commons.math3.random.RandomGenerator;

import monash.ml.model.Model;
import monash.ml.tools.StopWatch;
import weka.core.Utils;

public interface Evaluator {
		
	// --- --- --- CLI integration
	
	/** Returns the evaluator name */
	public String get_name();
	
	/** When the CLI parsing recognizes an evaluator, the remaining args are provided */
	public void init(String[] args) throws ParseException;
	
	// --- --- --- Evaluation
	
	/** Evaluate a model with an evaluator.
	 * The random number generator can be used by the evaluator, and will be passed to the model.*/
	public Evaluator.Result evaluate(Model model, RandomGenerator rg, SecureRandom rs, boolean verbose) throws Exception;
	
	
	/** Internal static class: evaluation result. */
	public static class Result {
		
		// --- --- --- Dataset
		public String datasetName;
		public int numAttribute = 0;
		public int numClass = 0;
		public int numData = 0;
		public int numTrain = 0;
		public int numTest = 0;
		
		// --- --- --- Classifier
		public String classifierName;

		// --- --- --- Measure
		double RMSE = 0;
		double errorRate = 0;
		
		// --- --- --- Timing
		StopWatch preprocessingTime, trainTime, testTime;
		
		public void preprocessBegins() {
			preprocessingTime.start();
		}
		
		public void preprocessStops() {
			preprocessingTime.stop();
		}
		
		public void trainBegins() {
			trainTime.start();
		}
		
		public void trainEnds() {
			trainTime.stop();
		}
		
		public void testBegins() {
			testTime.start();
		}
		
		public void testEnds() {
			testTime.stop();
		}
		
		public void setInfo(String classifierName, String datasetName, int numAttribute, int numClass, int numData) {
			this.classifierName = classifierName;
			this.datasetName = datasetName;
			this.numAttribute = numAttribute;
			this.numClass = numClass;
			this.numData = numData;
		}
		
		// --- --- --- Constructor
		public Result() {			
			this.classifierName = "Unspecified";
			
			this.datasetName = "Unspecified";
			this.numAttribute = 0;
			this.numClass = 0;
			this.numData = 0;
			
			this.preprocessingTime = new StopWatch();
			this.trainTime = new StopWatch();
			this.testTime = new StopWatch();
		}
		
		// --- --- --- Print the result
		public String toString() {
			StringBuilder sb = new StringBuilder();
			
			sb.append("\n--------------------- Data Information ----------------------");
			sb.append("\nDataset:        " + datasetName);
			sb.append("\nNb attributes:  " + numAttribute);
			sb.append("\nNb classes:     " + numClass);
			sb.append("\nNb data:        " + numData);
			sb.append("\nNb train data:  " + numTrain);
			sb.append("\nNb test data:   " + numTest);
			
			sb.append("\n\n--------------------- Results ----------------------");
			sb.append("\nClassifier:         " + classifierName);
			sb.append("\nPreprocessing time: " + preprocessingTime.toString());
			sb.append("\nTraining time:      " + trainTime.toString());
			sb.append("\nTesting time:       " + testTime.toString());
			sb.append("\nRMSE:               " + Utils.doubleToString(RMSE, 6, 4));
			sb.append("\nError:              " + Utils.doubleToString(errorRate, 6, 4));
			sb.append("\n");

			return sb.toString();
		}
	}
	
}

