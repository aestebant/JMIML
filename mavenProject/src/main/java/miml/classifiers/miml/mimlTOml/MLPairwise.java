package miml.classifiers.miml.mimlTOml;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.Pairwise;
import weka.classifiers.Classifier;
import weka.core.Instance;

public class MLPairwise extends Pairwise {
    public MLPairwise(Classifier classifier) {
        super(classifier);
        soft = false;
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] scores = calculateScores(instance);
        for (int i = 0; i < scores.length; i++) {
            scores[i] /= (numLabels - 1);
        }
        return new MultiLabelOutput(scores, 0.5);
    }
}
