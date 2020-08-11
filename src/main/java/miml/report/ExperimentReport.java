package miml.report;

import miml.core.ConfigParameters;
import miml.data.MIMLInstances;
import miml.evaluation.EvaluatorHoldout;
import miml.evaluation.IEvaluator;
import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.MacroAverageMeasure;
import mulan.evaluation.measure.Measure;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Date;
import java.util.List;

public class ExperimentReport extends BaseMIMLReport {
    /**
     * Read the holdout results and transform to CSV format.
     *
     * @param evaluator the evaluator
     * @throws Exception the exception
     */
    @Override
    public String toCSV(IEvaluator evaluator) throws Exception {
        if (!(evaluator instanceof EvaluatorHoldout)) {
            return "ERROR: evaluator not supported";
        }

        boolean writeHeader = true;

        File file = new File(filename);
        file.getParentFile().mkdirs();
        try {
            writeHeader = file.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }

        Evaluation evaluationHoldout = (Evaluation) evaluator.getEvaluation();
        MIMLInstances data = ((EvaluatorHoldout) evaluator).getData();

        StringBuilder result = new StringBuilder();
        String measureName;

        // All evaluator measures
        List<Measure> measures = evaluationHoldout.getMeasures();
        // Measures selected by user
        if (this.measures != null)
            measures = filterMeasures(measures);

        if (writeHeader) {
            if (ConfigParameters.getIsTransformation()) {
                // Write header
                result.append("Algorithm," + "Classifier," + "Transform method," + "Dataset," + "ConfigurationFile," + "Train_time_ms," + "Test_time_ms,");
            } else {
                // Write header
                result.append("Algorithm," + "Dataset," + "ConfigurationFile," + "Train_time_ms," + "Test_time_ms,");
            }
            // Write measure's names
            for (Measure m : measures) {
                measureName = m.getName();
                result.append(measureName).append(",");

                if (m instanceof MacroAverageMeasure && this.labels) {

                    for (int i = 0; i < data.getNumLabels(); i++) {
                        result.append(measureName).append("-").append(data.getDataSet().attribute(data.getLabelIndices()[i]).name()).append(",");
                    }
                }
            }
            result.append(System.getProperty("line.separator"));
        }

        if(ConfigParameters.getIsTransformation()) {
            result.append(ConfigParameters.getAlgorithmName()).append(",").append(ConfigParameters.getClassifierName()).append(",").append(ConfigParameters.getTransformationMethod()).append(",").append(ConfigParameters.getDataFileName()).append(",").append(ConfigParameters.getConfigFileName()).append(",").append(((EvaluatorHoldout) evaluator).getTrainTime()).append(",").append(((EvaluatorHoldout) evaluator).getTestTime()).append(",");
        } else {
            result.append(ConfigParameters.getAlgorithmName()).append(",").append(ConfigParameters.getDataFileName()).append(",").append(ConfigParameters.getConfigFileName()).append(",").append(((EvaluatorHoldout) evaluator).getTrainTime()).append(",").append(((EvaluatorHoldout) evaluator).getTestTime()).append(",");
        }

        // Write value for each measure
        for (Measure m : measures) {
            result.append(m.getValue()).append(",");
            if (m instanceof MacroAverageMeasure && this.labels) {
                for (int i = 0; i < data.getNumLabels(); i++) {
                    result.append(((MacroAverageMeasure) m).getValue(i)).append(",");
                }
            }

        }
        result.append(System.getProperty("line.separator"));

        Files.write(Paths.get(filename), result.toString().getBytes(), StandardOpenOption.APPEND);
        System.out.println("" + new Date() + ": " + "Experiment results saved in " + filename);

        return result.toString();
    }
}
