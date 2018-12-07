package monash.ml.evaluator;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.SecureRandom;

import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.random.RandomGenerator;

import monash.ml.model.Model;
import monash.ml.tools.arff.ArffFile;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * Holdout. Expects a ratio (to put in test) and one file
 */
public class TrainTest implements Evaluator {

	// --- ---- --- Constant
	public static final String name = "traintest";

	// --- --- --- Fields
	private Path train_file;
	private Path test_file;
	

	// --- --- --- Interface implementation

	@Override
	public String get_name() {
		return name;
	}

	@Override
	public void init(String[] args) throws ParseException {
		// --- --- --- Argument validation
		if (args.length != 2) {
			throw new ParseException("Evaluator " + name + " expects a training file and a test file");
		}
		// --- --- --- Is the training file readable?
		train_file = Paths.get(args[0]);
		if (!Files.isReadable(train_file)) {
			throw new ParseException("Evaluator " + name + " cannot read file `" + args[0] + "'");
		}
		// --- --- --- Is the file readable?
		test_file = Paths.get(args[1]);
		if (!Files.isReadable(test_file)) {
			throw new ParseException("Evaluator " + name + " cannot read file `" + args[1] + "'");
		}
	}

	@Override
	public Result evaluate(Model model, RandomGenerator rg, SecureRandom sr, boolean verbose) throws Exception {
		
		Result result = new Result();
		result.preprocessBegins();
		
		// --- --- --- Load file and prepare the split

		// Load train. Use train as the reference for the structure.
		// Load a file. Note: go over all the file to count the number of data!
		// ArffFiles hard-code the class as the last attribute. Use this structure.
		ArffFile trainFile = new ArffFile(train_file);
		Instances structure = trainFile.getStructure();
		
		// Initialisation of the result
		result.setInfo(model.getModelName(),
				train_file.getFileName().toString(),
				trainFile.getStructure().numAttributes(), trainFile.getNumClasses(), trainFile.getNumData());		
		
		result.numTrain = trainFile.getNumData();
			
		// Load test.
		ArffFile testFile = new ArffFile(test_file);
		result.numTest = testFile.getNumData();
		ArffReader testReader = testFile.getNewReader();
		
		result.preprocessStops();
		
	

		// --- --- --- Train
		if (verbose) {
			System.out.println("\n---------------------- Training Started ----------------------");
		}
		result.trainBegins();
		model.train(trainFile, rg);
		result.trainEnds();
		if (verbose) {
			System.out.println("\n---------------------- Training Finished ----------------------");
		}

		
		// --- --- --- Test
		if (verbose) {
			System.out.println("\n---------------------- Testing Started ----------------------");
		}
		result.testBegins();
		Instance current;
		int nError=0;
		while ((current = testReader.readInstance(structure)) != null) {
			double[] probs = model.distributionForInstance(current);
			int x_C = (int) current.classValue();
			// --- --- --- Update RMSE: accumulate here.
			int pred = -1;
			double bestProb = Double.MIN_VALUE;
			for (int y = 0; y < (result.numClass); y++) {
				if (!Double.isNaN(probs[y])) {
					if (probs[y] > bestProb) {
						pred = y;
						bestProb = probs[y];
					}
					result.RMSE += (1 / (double) (result.numClass) * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
				} else {
					System.err.println("probs[ " + y + "] is NaN! oh no!"); 
				}
			}
			// --- --- --- Update ERROR
			if (pred != x_C) {
				++nError;
			}
		}
		// Finish RMSE and error
		result.RMSE = Math.sqrt(result.RMSE / result.numTest);
		result.errorRate = nError / result.numTest;
		result.testEnds();
		if (verbose) {
			System.out.println("\n--------------------- Testing Finished ----------------------");
		}

		return result;
	}

}
