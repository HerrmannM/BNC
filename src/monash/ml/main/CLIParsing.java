package monash.ml.main;

import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import monash.ml.evaluator.Evaluator;
import monash.ml.model.Model;

public class CLIParsing {
	
	// --- --- --- Fields
	
	private final HashMap<String, Model> models;
	private final HashMap<String, Evaluator> evaluators;
	private final String[] args;
	private final Options options;
	
	private Model m;
	private Evaluator e;
	
	// --- --- --- Constructor
	public CLIParsing(String[] args, HashMap<String, Model> models, HashMap<String, Evaluator> evaluators) throws ParseException {
		this.args = args;
		this.models = models;
		this.evaluators = evaluators;
		this.options = getCLIOptions();
		doParsing();
	}
	
	// --- --- --- Access
	public Model getModel() {
		return m;
	}
	
	public Evaluator getEvaluator() {
		return e;
	}
	
	
	// --- --- --- Create options.
	
	/** The goal here is mainly to create switches allowing to read the model and evaluator names. */
	private Options getCLIOptions() {
		Options options = new Options();
		
		// --- Model
		String md_desc = "Specify the model to be used. Available: " + models.keySet();
		Option md_opt = Option.builder().longOpt("model").desc(md_desc).required()
				.hasArgs().argName("model name and arguments").build();
		
		// --- Evaluator
		String ev_desc = "Specify the evaluator to be used. Available: " + evaluators.keySet();
		Option ev_opt = Option.builder().longOpt("evaluator").desc(ev_desc).required()
				.hasArgs().argName("evaluator name and arguments").build();
		
		// --- Final construction
		options.addOption(md_opt).addOption(ev_opt);
		return options;
	}

	private void doParsing() {

		// Create the usage message
		String launchstr = "java classifier.jar";
		String header = "Arguments:";
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
			
			// Check for the model. Remove the 'name' from the list of args before initalisation.
			String[] md_args = commandLine.getOptionValues("model");
			if(md_args.length == 0) {
				throw new ParseException("The --model option requires at least a model name");
			}
			m = models.get(md_args[0]);
			if (m == null) {
				throw new ParseException("Unrecognized model `"+md_args[0]+"'");
			}
			m.init(Arrays.copyOfRange(md_args, 1, md_args.length));
			
			// Check for the evaluator Remove the 'name' from the list of args before initalisation.
			String[] ev_args = commandLine.getOptionValues("evaluator");
			if(ev_args.length == 0) {
				throw new ParseException("The --evaluator option requires at least an evaluator name");
			}
			e = evaluators.get(ev_args[0]);
			if (e == null) {
				throw new ParseException("Unrecognized evaluator `"+ev_args[0]+"'");
			}
			e.init(Arrays.copyOfRange(ev_args, 1, ev_args.length));
			
		} catch (ParseException exception) {
			System.err.print("Command line arguments: ");
			System.err.println(exception.getMessage());
			formatter.printHelp(launchstr, header, options, footer, true);
			System.exit(2);
		}
	}
}
