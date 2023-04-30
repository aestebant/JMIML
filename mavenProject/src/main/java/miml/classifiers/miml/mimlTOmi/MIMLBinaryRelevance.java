package miml.classifiers.miml.mimlTOmi;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper for mulan BinaryRelevance to be used in MIML to MI algorithms.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public class MIMLBinaryRelevance extends BinaryRelevance {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = 1706817441965109002L;
	protected List<Classifier> ensemble;
	private BinaryRelevanceTransformation brt;

	/**
     * Creates a new instance.
     *
     * @param classifier The base-level classification algorithm that will be
     * used for training each of the binary models.
     */
	public MIMLBinaryRelevance(Classifier classifier) {
		super(classifier);
		// TODO Auto-generated constructor stub
	}

	protected void buildInternal(MultiLabelInstances train) throws Exception {
		ensemble = new ArrayList<>(numLabels);

		debug("preparing shell");
		brt = new BinaryRelevanceTransformation(train);

		for (int i = 0; i < numLabels; i++) {
			ensemble.add(AbstractClassifier.makeCopy(baseClassifier));
			Instances shell = brt.transformInstances(i);
			debug("Bulding model " + (i + 1) + "/" + numLabels);
			ensemble.get(i).buildClassifier(shell);
			debug(ensemble.get(i).toString());
		}
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance) {
		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		for (int counter = 0; counter < numLabels; counter++) {
				Instance transformedInstance = brt.transformInstance(instance, counter);
				double[] distribution = null;
				try {
					// AQUÍ SI EL CLASIFICADOR NO HA GENERADO MODELO PARA COUNTER SE IGNORARÍA ESTA DISTRIBUCIÓN
					distribution = ensemble.get(counter).distributionForInstance(transformedInstance);
				} catch (Exception ignored) {
				}
				if (distribution != null) {
					int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;
					// Ensure correct predictions both for class values {0,1} and {1,0}
					bipartition[counter] = maxIndex == 1;
					// The confidence of the label being equal to 1
					confidences[counter] = distribution[1];
				}
		}

		return new MultiLabelOutput(bipartition, confidences);
	}
}
