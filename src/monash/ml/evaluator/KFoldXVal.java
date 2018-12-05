package monash.ml.evaluator;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.SecureRandom;
import java.util.ArrayList;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.random.RandomGenerator;

import monash.ml.model.Model;
import monash.ml.tools.arff.ArffFile;
import monash.ml.tools.arff.ArffFileJoin;
import monash.ml.tools.arff.Utility;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class KFoldXVal implements Evaluator {
	
	// --- ---- --- Constant
	public static final String name = "kfoldxval";
	
	// --- --- --- Fields
	private Path input_file;
	private int kfold;
	private int expRound;

	// --- --- --- Interface implementation
	@Override
	public String get_name() {
		return name;
	}

	@Override
	public void init(String[] args) throws ParseException {
		// --- --- --- Argument validation
		if (args.length != 3) {
			throw new ParseException("Evaluator " + name + " expects a kfold cross evaluation indice, a number of rounds and a file");
		}
		
		// --- --- --- Read the k
		try {
			this.kfold = Integer.parseInt(args[0]);
		} catch (NumberFormatException e) {
			throw new ParseException("Evaluator " + name + " cannot read the k indice (found `" + args[0] + "'");
		}
		
		// --- --- --- Read the number of rounds
		try {
			this.expRound = Integer.parseInt(args[1]);
		} catch (NumberFormatException e) {
			throw new ParseException("Evaluator " + name + " cannot read the number of rounds (found `" + args[1] + "'");
		}

		// --- --- --- Is the file readable?
		input_file = Paths.get(args[2]);
		if (!Files.isReadable(input_file)) {
			throw new ParseException("Evaluator " + name + " cannot read file `" + args[2] + "'");
		}
	}

	@Override
	public Result evaluate(Model model, RandomGenerator rg, SecureRandom sr, boolean verbose) throws Exception {
		
		Result result = new Result();
		result.preprocessBegins();
		
		// Load a file. Note: go over all the file to count the number of data!
		// ArffFile hardcode the class as the last attribute. Use this structure.
		ArffFile arffFile = new ArffFile(input_file);
		Instances structure = arffFile.getStructure();
		
	
		// Initialisation of the result
		result.setInfo(model.getModelName(),
				input_file.getFileName().toString(),
				arffFile.getStructure().numAttributes(), arffFile.getNumClasses(), arffFile.getNumData());
		result.preprocessStops();
		

		
		// --- Do several round of KFold cross validation
		for (int exp = 0; exp < expRound; exp++) {
			double expRMSE = 0.0;
			double expErrorRate = 0.0;
			int expNErrors = 0;
			
			if (verbose) {
				System.out.println("----------------- Round number " + exp + "----------------------");
			}
			
			// Do the partition for the current round
			result.preprocessBegins();
			ArffFile[] chunks = Utility.partitionData(arffFile, kfold, sr);
			result.preprocessStops();
			
			// Do the cross validation
			for(int testFold=0; testFold<kfold; testFold++) {
				
				// --- --- --- Prepare the set of data
				result.preprocessBegins();
			
				// --- Train on all but the current K

				ArrayList<ArffFile> trainFilesAL= new ArrayList<>(kfold);
				for(int i=0; i<kfold; ++i) {
					if(i!=testFold) {
						trainFilesAL.add(chunks[i]);
					}
				}
				ArffFile[] trainFiles = trainFilesAL.toArray(new ArffFile[kfold-1]);
				ArffFileJoin trainReaderFactory = new ArffFileJoin(trainFiles);
				result.numTrain += trainReaderFactory.getNumData();
				
				// --- Test on the current K
				ArffReader testReader = chunks[testFold].getNewReader();
				result.numTest += chunks[testFold].getNumData();
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
				while ((current = testReader.readInstance(structure)) != null) {
					double[] probs = model.distributionForInstance(current);
					int x_C = (int) current.classValue();
					// --- --- --- Update RMSE: accumulate here.
					int pred = -1;
					double bestProb = -1.0;
					for (int y = 0; y < result.numClass; y++) {
						if (!Double.isNaN(probs[y])) {
							if (probs[y] > bestProb) {
								pred = y;
								bestProb = probs[y];
							}
							expRMSE += (1 / (double) (result.numClass) * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
						} else {
							System.err.println("probs[ " + y + "] is NaN! oh no!");
						}
					}
					// --- --- --- Update ERROR
					if (pred != x_C) {
						expNErrors += 1;
					}
				}
				result.testEnds();
				
				
			}
			expRMSE = Math.sqrt(expRMSE / result.numData);
			expErrorRate =  ((double)expNErrors)/result.numData;
			result.RMSE += expRMSE;
			result.errorRate +=expErrorRate;
			if (verbose) {
				System.out.println("\n--------------------- Testing Finished ----------------------");
				System.out.println("RMSE after exp " + exp+ ": " + result.RMSE / (exp+1));
				System.out.println("Error after exp " + exp + ": " + result.errorRate / (exp+1));
			}
		}
		
		// Finish RMSE and error
		result.RMSE = result.RMSE/expRound;
		result.errorRate = result.errorRate / expRound;
		
		return result;
	}

}
