/*    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

package es.uco.kdis.miml.evaluation;

import java.util.Date;

import org.apache.commons.configuration2.Configuration;

import es.uco.kdis.miml.classifiers.miml.IMIMLClassifier;
import es.uco.kdis.miml.core.IConfiguration;
import es.uco.kdis.miml.data.MIMLInstances;
import mulan.data.InvalidDataFormatException;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

/**
 * Class that allow evaluate an algorithm applying a cross-validation method
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180630
 */
public class EvaluatorCV implements IConfiguration, IEvaluator<MultipleEvaluation> {

	/** The evaluation. */
	private MultipleEvaluation evaluation;

	/** The data. */
	private MIMLInstances data;

	/** The number of folds. */
	private int numFolds;
	
	/** The seed for the partition */
	protected int seed = 1;

	/**
	 * Instantiates a new Holdout evaluator.
	 *
	 * @param data     the data used in the experiment
	 * @param numFolds the number of folds used in the cross-validation
	 */
	public EvaluatorCV(MIMLInstances data, int numFolds) {
		this.data = data;
		this.numFolds = numFolds;
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public EvaluatorCV() {

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#runExperiment(mimlclassifier.MIMLClassifier)
	 */
	@Override
	public void runExperiment(IMIMLClassifier classifier) {
		Evaluator eval = new Evaluator();
		eval.setSeed(seed);
		System.out.println("" + new Date() + ": " + "Initializing cross validation");
		evaluation = eval.crossValidate(classifier, data, numFolds);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#getEvaluation()
	 */
	@Override
	public MultipleEvaluation getEvaluation() {
		return evaluation;
	}

	/**
	 * Gets the data used for evaluate the measures.
	 *
	 * @return the data
	 */
	public MIMLInstances getData() {
		return data;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {

		numFolds = configuration.getInt("numFolds", 5);
		seed = configuration.getInt("seed", 1);
		
		String arffFile = configuration.subset("data").getString("file");
		String xmlFileName = configuration.subset("data").getString("xmlFile");

		try {
			data = new MIMLInstances(arffFile, xmlFileName);
		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		}

	}

}