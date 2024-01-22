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

package miml.evaluation;

import miml.classifiers.miml.IMIMLClassifier;
import miml.classifiers.miml.mimlTOml.MIMLClassifierToML;
import miml.classifiers.ml.RFPCT;
import miml.core.ConfigParameters;
import miml.core.IConfiguration;
import miml.data.MIMLInstances;
import mulan.classifier.MultiLabelOutput;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.*;
import org.apache.commons.configuration2.Configuration;
import weka.core.Instances;
import weka.filters.Filter;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Class that allow evaluate an algorithm applying a holdout method.
 *
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180630
 */
public class EvaluatorHoldout implements IConfiguration, IEvaluator<Evaluation> {

	/** The evaluation method used in holdout. */
	protected Evaluation evaluation;

	/** The data used in the experiment. */
	protected MIMLInstances trainData;

	/** The test data used in the experiment. */
	protected MIMLInstances testData;

	/** Train time in milliseconds. */
	protected long trainTime;

	/** Test time in milliseconds. */
	protected long testTime;

	protected String clusWorkingDir;
	protected String clusDataset;

	/**
	 * Instantiates a new holdout evaluator with provided train and test partitions.
	 *
	 * @param trainData The train data used in the experiment.
	 * @param testData  The test data used in the experiment.
	 * @throws InvalidDataFormatException To be handled.
	 */
	public EvaluatorHoldout(MIMLInstances trainData, MIMLInstances testData) throws InvalidDataFormatException {
		this.trainData = trainData;
		this.testData = testData;
	}

	/**
	 * Instantiates a new holdout evaluator with random partitioning method.
	 *
	 * @param mimlDataSet     The dataset to be used.
	 * @param percentageTrain The percentage of train.
	 * @throws Exception If occur an error during holdout experiment.
	 */
	public EvaluatorHoldout(MIMLInstances mimlDataSet, double percentageTrain) throws Exception {
		this(mimlDataSet, percentageTrain, 1, 1);
	}

	/**
	 * Instantiates a new Holdout evaluator with a partitioning method and a seed.
	 *
	 * @param mimlDataSet     The dataset to be used.
	 * @param percentageTrain The percentage of train.
	 * @param seed Seed for randomization.
	 * @param method partitioning method.
	 * <ul>
	 * <li> 1 random partitioning </li>
	 * <li> 2 powerset partitioning </li>
	 * <li> 3 iterative partitioning </li>
	 * </ul>
	 * @throws Exception If occur an error during holdout experiment.
	 */
	public EvaluatorHoldout(MIMLInstances mimlDataSet, double percentageTrain, int seed, int method) throws Exception {

		List<MIMLInstances> list = MIMLInstances.splitData(mimlDataSet, percentageTrain, seed, method);
		this.trainData = list.get(0);
		this.testData = list.get(1);
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public EvaluatorHoldout() {

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#runExperiment(mimlclassifier.MIMLClassifier)
	 */
	@Override
	public void runExperiment(IMIMLClassifier classifier) {
		Evaluator eval = new Evaluator();
		System.out.println(new Date() + ": " + "Building model");
		if (classifier instanceof MIMLClassifierToML && ((MIMLClassifierToML) classifier).baseClassifier instanceof RFPCT) {
			((RFPCT) ((MIMLClassifierToML) classifier).baseClassifier).setClusWorkingDir(clusWorkingDir);
			((RFPCT) ((MIMLClassifierToML) classifier).baseClassifier).setDatasetName(clusDataset);
		}
		long startTime = System.nanoTime();
        try {
            classifier.build(trainData);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        long estimatedTime = System.nanoTime() - startTime;
		trainTime = TimeUnit.NANOSECONDS.toMillis(estimatedTime);

		if (classifier instanceof MIMLClassifierToML) {
			MIMLClassifierToML mlClassifier = (MIMLClassifierToML) classifier;
			if (mlClassifier.baseClassifier instanceof RFPCT) {
				Instances auxTestInstances;
                try {
					MultiLabelInstances auxTestData = mlClassifier.transformationMethod.transformDataset(testData);
					auxTestInstances = Filter.useFilter(auxTestData.getDataSet(), mlClassifier.removeFilter);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                try {
                    testData = new MIMLInstances(auxTestInstances, testData.getLabelsMetaData());
                } catch (InvalidDataFormatException e) {
                    throw new RuntimeException(e);
                }
            }
		}

		System.out.println(new Date() + ": " + "Getting evaluation results");
		startTime = System.nanoTime();
		if (classifier instanceof MIMLClassifierToML && ((MIMLClassifierToML) classifier).baseClassifier instanceof RFPCT) {
			List<Measure> measures = new ArrayList<>(26);
			measures.add(new HammingLoss());
			measures.add(new SubsetAccuracy());
			measures.add(new ExampleBasedPrecision());
			measures.add(new ExampleBasedRecall());
			measures.add(new ExampleBasedFMeasure());
			measures.add(new ExampleBasedAccuracy());
			measures.add(new ExampleBasedSpecificity());
			measures.add(new MicroPrecision(trainData.getNumLabels()));
			measures.add(new MicroRecall(trainData.getNumLabels()));
			measures.add(new MicroFMeasure(trainData.getNumLabels()));
			measures.add(new MicroSpecificity(trainData.getNumLabels()));
			measures.add(new MacroPrecision(trainData.getNumLabels()));
			measures.add(new MacroRecall(trainData.getNumLabels()));
			measures.add(new MacroFMeasure(trainData.getNumLabels()));
			measures.add(new MacroSpecificity(trainData.getNumLabels()));
			measures.add(new AveragePrecision());
			measures.add(new Coverage());
			measures.add(new OneError());
			measures.add(new IsError());
			measures.add(new ErrorSetSize());
			measures.add(new RankingLoss());
			measures.add(new MeanAveragePrecision(trainData.getNumLabels()));
			measures.add(new GeometricMeanAveragePrecision(trainData.getNumLabels()));
			measures.add(new MeanAverageInterpolatedPrecision(trainData.getNumLabels(), 10));
			measures.add(new GeometricMeanAverageInterpolatedPrecision(trainData.getNumLabels(), 10));
			measures.add(new LogLoss());

            try {
                eval.evaluate(((MIMLClassifierToML) classifier).baseClassifier, this.testData, measures);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            LabelMatrix lm;
            try {
                lm = getLabelsClus(this.testData.getNumInstances(), this.trainData.getNumLabels());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            try {
                this.evaluation = evaluate(lm.realLabels, lm.predLabels, measures);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else {
			try {
				evaluation = eval.evaluate(classifier, testData, trainData);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}
		estimatedTime = System.nanoTime() - startTime;
		testTime = TimeUnit.NANOSECONDS.toMillis(estimatedTime);
	}

	private LabelMatrix getLabelsClus(int nInstances, int nLabels) throws IOException {
		LabelMatrix lm = new LabelMatrix(nInstances, nLabels);
		Path filePath = Paths.get(clusWorkingDir, clusDataset +"-train.arff");
		Instances inst = new Instances(new FileReader(filePath.toString()));
		for(int i = 0; i < nInstances; ++i) {
			for(int l = 0; l < nLabels; ++l) {
				int v = (int)inst.get(i).value(l);
				if (v == 1) {
					lm.realLabels[i][l] = 0;
				} else {
					lm.realLabels[i][l] = 1;
				}
				v = (int)inst.get(i).value(l + nLabels);
				if (v == 1) {
					lm.predLabels[i][l] = 0;
				} else {
					lm.predLabels[i][l] = 1;
				}
			}
		}
		return lm;
	}

	private static Evaluation evaluate(int[][] groundTruth, int[][] predictedLabels, List<Measure> measures) throws Exception {
        for (Measure m : measures) {
            m.reset();
        }
		int numLabels = groundTruth[0].length;
		Set<Measure> failed = new HashSet<>();
		int numInstances = groundTruth.length;

		for(int i = 0; i < numInstances; ++i) {
			boolean[] predBool = new boolean[numLabels];
			boolean[] realBool = new boolean[numLabels];
			for(int j = 0; j < numLabels; ++j) {
                predBool[j] = predictedLabels[i][j] == 1;
                realBool[j] = groundTruth[i][j] == 1;
			}
			MultiLabelOutput output2 = new MultiLabelOutput(predBool);
			GroundTruth truth2 = new GroundTruth(realBool);

            for (Measure m : measures) {
                if (!failed.contains(m)) {
                    try {
                        m.update(output2, truth2);
                    } catch (Exception var14) {
                        failed.add(m);
                    }
                }
            }
		}
		return new Evaluation(measures, null);
	}


	/**
	 * Gets the time spent in training.
	 *
	 * @return The train time.
	 */
	public long getTrainTime() {
		return trainTime;
	}

	/**
	 * Gets the time spent in testing.
	 *
	 * @return The test time.
	 */
	public long getTestTime() {
		return testTime;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#getEvaluation()
	 */
	@Override
	public Evaluation getEvaluation() {
		return evaluation;
	}

	/**
	 * Gets the data used in the experiment.
	 *
	 * @return The data.
	 */
	public MIMLInstances getData() {
		return testData;
	}


	/*
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {
		String arffFileTrain = configuration.subset("data").getString("trainFile");
		String xmlFileName = configuration.subset("data").getString("xmlFile");
		String arffFileTest = configuration.subset("data").getString("testFile");

		if (arffFileTest == null) {
			// if partitioning method is not provided it will be random partitioning
			String partitioning = configuration.getString("partitionMethod", "random");
			int method;
			switch (partitioning) {
				case "powerset":
					method = 2;
					break;
				case "iterative":
					method = 3;
					break;
				default:
					method = 1; // by default random partitioning
			}
			int seed = configuration.getInt("seed", 1);
			double percentageTrain = configuration.getDouble("percentageTrain", 80);
            List<MIMLInstances> list;
            try {
                list = MIMLInstances.splitData(new MIMLInstances(arffFileTrain, xmlFileName), percentageTrain, seed, method);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            this.trainData = list.get(0);
			this.testData = list.get(1);
		} else {
            try {
                trainData = new MIMLInstances(arffFileTrain, xmlFileName);
				testData = new MIMLInstances(arffFileTest, xmlFileName);
            } catch (InvalidDataFormatException e) {
                throw new RuntimeException(e);
            }
		}
		this.clusWorkingDir = configuration.getString("clusWorkingDir", "clusFolder");
		this.clusDataset = configuration.getString("clusDataset", "clusDataset");
		ConfigParameters.setDataFileName(new File(arffFileTrain).getName());
	}

	private static class LabelMatrix {
		public int[][] realLabels;
		public int[][] predLabels;

		public LabelMatrix(int nInstances, int nLabels) {
			this.realLabels = new int[nInstances][nLabels];
			this.predLabels = new int[nInstances][nLabels];
		}
	}
}
