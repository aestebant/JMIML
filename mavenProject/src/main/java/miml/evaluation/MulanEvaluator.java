package miml.evaluation;

import miml.classifiers.ml.RFPCT;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.Measure;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.*;
import clus.Clus;

public class MulanEvaluator extends Evaluator {
    public Evaluation evaluate(RFPCT rfpct, MultiLabelInstances testData, List<Measure> measures) throws Exception {
        boolean isEnsemble = rfpct.isEnsemble;
        boolean isRuleBased = rfpct.isRuleBased;
        boolean isRegression;
        MultiLabelOutput output = rfpct.makePrediction(testData.getDataSet().instance(0));
        isRegression = output.hasPvalues();

        // write the supplied MultilabelInstances object in an arff formated file (accepted by CLUS)
        RFPCT.makeClusCompliant(testData, Paths.get(rfpct.getClusWorkingDir(), rfpct.getDatasetName() + "-test.arff").toString());

        // call Clus.main to write the output files!
        ArrayList<String> clusArgsList = new ArrayList<>();
        if (isEnsemble) {
            clusArgsList.add("-forest");
        }
        if (isRuleBased) {
            clusArgsList.add("-rules");
        }
        // the next argument passed to Clus is the settings file!
        clusArgsList.add(Paths.get(rfpct.getClusWorkingDir(), rfpct.getDatasetName() + "-train.s").toString());
        String[] clusArgs = clusArgsList.toArray(new String[0]);
        Clus.main(clusArgs);

        // then parse the output files and finally update the measures!
        // open and load the test set predictions file, which is in arff format
        BufferedReader reader = new BufferedReader(new FileReader(Paths.get(rfpct.getClusWorkingDir(), rfpct.getDatasetName() + "-train.test.pred.arff").toString()));
        Instances predictionInstances = new Instances(reader);
        reader.close();

        checkLearner(rfpct);
        checkData(testData);
        checkMeasures(measures);

        // reset measures
        for (Measure m : measures) m.reset();

        int numLabels = testData.getNumLabels();
        Set<Measure> failed = new HashSet<>();
        Instances testDataset = testData.getDataSet();

        int numInstances = testDataset.numInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testDataset.instance(instanceIndex);
            if (testData.hasMissingLabels(instance)) continue;
            Instance labelsMissing = (Instance) instance.copy();
            labelsMissing.setDataset(instance.dataset());
            for (int i = 0; i < testData.getNumLabels(); i++) {
                labelsMissing.setMissing(testData.getLabelIndices()[i]);
            }
            GroundTruth truth;
            boolean[] trueLabels = new boolean[numLabels];
            double[] trueValues = new double[numLabels];
            // clus way
            Instance predictionInstance = predictionInstances.instance(instanceIndex);
            double[] predictionsPerSample = new double[testData.getNumLabels()];
            int k = 0;
            for (int j = 0; j < predictionInstance.numValues() - 1; j++) {
                String pred = predictionInstance.toString(j);
                // collect the ground truth
                if (j < testData.getNumLabels()) {
                    if (isRegression) trueValues[j] = Double.parseDouble(pred);
                    else trueLabels[j] = Double.parseDouble(pred) > 0.5;
                }
                // collect predicted values
                if (isRegression) {
                    if (isEnsemble && !isRuleBased) {
                        if (j >= (testData.getNumLabels())) {
                            predictionsPerSample[k] = predictionInstance.value(j);
                            k++;
                        }
                    } else {
                        if (j >= (testData.getNumLabels() * 2 + 1)) {
                            predictionsPerSample[k] = predictionInstance.value(j);
                            k++;
                        }
                    }
                } else {
                    if (isEnsemble && !isRuleBased) {
                        if (j >= testData.getNumLabels() * 2) {
                            predictionsPerSample[k] = predictionInstance.value(j);
                            j++;
                            k++;
                        }
                    } else {
                        if (j >= (testData.getNumLabels() * 5 + 1)) {
                            predictionsPerSample[k] = predictionInstance.value(j) / (predictionInstance.value(j) + predictionInstance.value(j + 1));
                            j++;
                            k++;
                        }
                    }
                }
                if (k == testData.getNumLabels()) break;
            }
            if (!isRegression) {
                output = new MultiLabelOutput(predictionsPerSample, 0.5);
                truth = new GroundTruth(trueLabels);
            } else {
                output = new MultiLabelOutput(predictionsPerSample, true);
                truth = new GroundTruth(trueValues);
            }
            for (Measure m : measures) {
                if (!failed.contains(m)) {
                    try {
                        m.update(output, truth);
                    } catch (Exception ex) {
                        failed.add(m);
                    }
                }
            }
        }
        return new Evaluation(measures, testData);
    }

    private void checkLearner(MultiLabelLearner learner) {
        if (learner == null) throw new IllegalArgumentException("Learner to be evaluated is null.");
    }

    private void checkData(MultiLabelInstances data) {
        if (data == null) throw new IllegalArgumentException("Evaluation data object is null.");
    }

    private void checkMeasures(List<Measure> measures) {
        if (measures == null) throw new IllegalArgumentException("List of evaluation measures to compute is null.");
    }
}
