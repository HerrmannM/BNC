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
public class Holdout implements Evaluator {

	// --- ---- --- Constant
	public static final String name = "holdout";

	// --- --- --- Fields
	private Path input_file;
	private double ratio;

	// --- --- --- Interface implementation

	@Override
	public String get_name() {
		return name;
	}

	@Override
	public void init(String[] args) throws ParseException {
		// --- --- --- Argument validation
		if (args.length != 2) {
			throw new ParseException("Evaluator " + name + " expects a ratio (how many instances to put in the test) and a file");
		}
		// --- --- --- Read the ratio
		try {
			this.ratio = Double.parseDouble(args[0]);
		} catch (NumberFormatException e) {
			throw new ParseException("Evaluator " + name + " cannot read the ratio (found `" + args[0] + "'");
		}

		// --- --- --- Is the file readable?
		input_file = Paths.get(args[1]);
		if (!Files.isReadable(input_file)) {
			throw new ParseException("Evaluator " + name + " cannot read file `" + args[1] + "'");
		}
	}

	@Override
	public Result evaluate(Model model, RandomGenerator rg, SecureRandom sr, boolean verbose) throws Exception {
		
		Result result = new Result();
		result.preprocessBegins();
		
		// --- --- --- Load file and prepare the split

		// Load a file. Note: go over all the file to count the number of data!
		// ArffFile hardcode the class as the last attribute. Use this structure.
		ArffFile arffFile = new ArffFile(input_file);
		Instances structure = arffFile.getStructure();
		
		// Initialisation of the result
		result.setInfo(model.getModelName(),
				input_file.getFileName().toString(),
				arffFile.getStructure().numAttributes(), arffFile.getNumClasses(), arffFile.getNumData());

		// Split the file between train and test following the ratio
		ArffFile[] split = monash.ml.tools.arff.Utility.splitData(arffFile, ratio, sr);
		
		ArffFile trainReaderFactory = split[0];
		result.numTrain = trainReaderFactory.getNumData();
		
		ArffFile testReaderFactory = split[1];
		ArffReader testReader = testReaderFactory.getNewReader();
		result.numTest = testReaderFactory.getNumData();
		
		result.preprocessStops();
		
	

		// --- --- --- Train
		if (verbose) {
			System.out.println("\n---------------------- Training Started ----------------------");
		}
		result.trainBegins();
		model.train(trainReaderFactory, rg);
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
