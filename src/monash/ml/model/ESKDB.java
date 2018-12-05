package monash.ml.model;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.random.RandomGenerator;

import monash.ml.model.Model;
import monash.ml.model.eskdb.wdBayesOnlinePYP;
import monash.ml.tools.arff.ArffReaderFactory;
import weka.core.Instance;

public final class ESKDB implements Model {

	// --- ---- --- Constant
	public static final String name = "ESKDB";

	// --- --- --- Fields
	private wdBayesOnlinePYP backendModel;
	private String[] args;
	private Options options;
	private int depth_k;
	private int tying;
	private int iterGibbs;
	private int ensemble;
	private boolean mestimation;
	private boolean backoff;

	// --- --- --- Model interface

	@Override
	public String getModelName() {
		return name;
	}

	@Override
	public void init(String[] args) throws ParseException {
		// --- Read the arguments
		this.args = args;
		this.options = getCLIOptions();
		doParsing();
	}

	@Override
	public void train(ArffReaderFactory readerFactory, RandomGenerator rg) throws Exception {
		// --- Create a new backend each time!
		backendModel = new wdBayesOnlinePYP();
		backendModel.set_m_S(name);
		backendModel.setK(depth_k);
		backendModel.setMEstimation(mestimation);
		backendModel.setGibbsIteration(iterGibbs);
		backendModel.setEnsembleSize(ensemble);
		backendModel.setBackoff(backoff);
		backendModel.setM_Tying(tying);
		backendModel.setPrint(true);
		backendModel.setRandomGenerator(rg);
		backendModel.buildClassifier(readerFactory);
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		return backendModel.distributionForInstance(instance);
	}

	// --- --- --- Create options.

	/** CLI Options for SKDB */
	private Options getCLIOptions() {
		Options options = new Options();
		
		// Specify the backoff.
		Option back = Option.builder("B").desc("Use backoff").build();
		
		// Specify the IterGibbs. Required.
		Option ensemble = Option.builder("E").desc("Specify the ensemble size of the model (integer value expected)").required()
				.hasArg().argName("ensemble size").build();
		
		// Specify the IterGibbs. Required.
		Option igibbs = Option.builder("I").desc("Specify the itergibss of the model (integer value expected)").required()
				.hasArg().argName("iteration gibbs").build();

		// Specify the depth of the model. Required.
		Option K = Option.builder("K").desc("Specify the depth of the model (integer value expected)").required()
				.hasArg().argName("model depth").build();

		// Specify the Tying. Required.
		Option tying = Option.builder("L").desc("Specify the tying value (integer value expected").required()
				.hasArg().argName("tying value").build();

		// Specify the smoothing method. If not set, use HDP
		Option smoothing = Option.builder("M").desc("Use MEstimation instead of HDP").build();

		// --- Final construction
		options.addOption(back).addOption(ensemble).addOption(igibbs).addOption(K).addOption(tying).addOption(smoothing);
		return options;
	}
	
	/** Do the CLI parsing */
	private void doParsing() {

		// Create the usage message
		String launchstr = "--model SKDB";
		String header = "SKDB Arguments:";
		String footer = "Monash Classifiers - Monash University, Melbourne, Australia";
		HelpFormatter formatter = new HelpFormatter();
		formatter.setWidth(120);

		// Print usage if no args
		if (args.length == 0) {
			formatter.printHelp(launchstr, header, options, footer, true);
			System.exit(1);
		}

		// Do the parsing
		CommandLineParser parser = new DefaultParser();
		CommandLine commandLine;

		try {
			commandLine = parser.parse(options, args);
			
			// Check for the backoff
			backoff = commandLine.hasOption("B");
			
			// Check the ensemble 'E' (required, so is present)
			try{
				ensemble = Integer.parseInt(commandLine.getOptionValue("E"));
			} catch (NumberFormatException e) {
				throw new ParseException("The -E option requires a positive integer value.");
			}
			if(ensemble<=0) {
				throw new ParseException("The -E option requires a positive integer value.");
			}
			
			// Check the gibbs 'I' (required, so is present)
			try{
				iterGibbs = Integer.parseInt(commandLine.getOptionValue("I"));
			} catch (NumberFormatException e) {
				throw new ParseException("The -I option requires a positive integer value.");
			}
			if(iterGibbs<=0) {
				throw new ParseException("The -I option requires a positive integer value.");
			}
			
			// Check the depth 'K' (required, so is present)
			try{
				depth_k = Integer.parseInt(commandLine.getOptionValue("K"));
			} catch (NumberFormatException e) {
				throw new ParseException("The -K option requires a positive integer value.");
			}
			if(depth_k<=0) {
				throw new ParseException("The -K option requires a positive integer value.");
			}
			
			// Check the tying 'L'
			if(commandLine.hasOption("L")) {
				try{
					tying = Integer.parseInt(commandLine.getOptionValue("L"));
				} catch (NumberFormatException e) {
					throw new ParseException("The -L option requires a positive integer value.");
				}
				if(tying<=0) {
					throw new ParseException("The -L option requires a positive integer value.");
				}
			} else {
				tying = 2;
			}
			
			// Check for the smoothing. false = HDP. true=MESTIMATION.
			mestimation = commandLine.hasOption("M");

		} catch (ParseException exception) {
			System.err.print("Command line arguments: ");
			System.err.println(exception.getMessage());
			formatter.printHelp(launchstr, header, options, footer, true);
			System.exit(2);
		}
	}

}
