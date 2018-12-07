package bnc.model;

import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.random.RandomGenerator;

import mltools.arff.ArffReaderFactory;
import weka.core.Instance;

public interface Model {
	
	// --- --- --- CLI integration
	
	/** Returns the model name */
	public String getModelName();

	/** When the CLI parsing recognizes an evaluator, the remaining args are provided */
	public void init(String[] args) throws ParseException;
	
	// --- --- --- Training
	
	/** Evaluate a model. A random number generator is passed as a parameter. */
	public void train(ArffReaderFactory readerFactory, RandomGenerator rg) throws Exception;
	
	// --- --- --- Querying
	/** Query the probability distribution for an instance */
	public double[] distributionForInstance(Instance instance);
	
}
