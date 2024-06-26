package miml.classifiers.mi;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

/**
 * <!-- globalinfo-start --> Reduces MI data into mono-instance data.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -M [1|2|3]
 *  The method used in transformation:
 *  1.arithmatic average; 2.geometric centor;
 *  3.using minimax combined features of a bag (default: 1)
 *
 *  Method 3:
 *  Define s to be the vector of the coordinate-wise maxima
 *  and minima of X, ie.,
 *  s(X)=(minx1, ..., minxm, maxx1, ...,maxxm), transform
 *  the exemplars into mono-instance which contains attributes
 *  s(X)
 * </pre>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <pre>
 * -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.rules.ZeroR)
 * </pre>
 *
 * <pre>
 * Options specific to classifier weka.classifiers.rules.ZeroR:
 * </pre>
 *
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @author Lin Dong (ld21@cs.waikato.ac.nz)
 * @version $Revision: 10369 $
 */
public class SimpleMI extends SingleClassifierEnhancer implements
		OptionHandler, MultiInstanceCapabilitiesHandler {

	/** for serialization */
	static final long serialVersionUID = 9137795893666592662L;

	/** arithmetic average */
	public static final int TRANSFORMMETHOD_ARITHMETIC = 1;
	/** geometric average */
	public static final int TRANSFORMMETHOD_GEOMETRIC = 2;
	/** using minimax combined features of a bag */
	public static final int TRANSFORMMETHOD_MINIMAX = 3;
	/** the transformation methods */
	public static final Tag[] TAGS_TRANSFORMMETHOD = {
			new Tag(TRANSFORMMETHOD_ARITHMETIC, "arithmetic average"),
			new Tag(TRANSFORMMETHOD_GEOMETRIC, "geometric average"),
			new Tag(TRANSFORMMETHOD_MINIMAX, "using minimax combined features of a bag") };

	/** the method used in transformation */
	protected int m_TransformMethod = TRANSFORMMETHOD_ARITHMETIC;

	/**
	 * Returns a string describing this filter
	 *
	 * @return a description of the filter suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Reduces MI data into mono-instance data.";
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tThe method used in transformation:\n"
				+ "\t1.arithmatic average; 2.geometric centor;\n"
				+ "\t3.using minimax combined features of a bag (default: 1)\n\n"
				+ "\tMethod 3:\n"
				+ "\tDefine s to be the vector of the coordinate-wise maxima\n"
				+ "\tand minima of X, ie., \n"
				+ "\ts(X)=(minx1, ..., minxm, maxx1, ...,maxxm), transform\n"
				+ "\tthe exemplars into mono-instance which contains attributes\n"
				+ "\ts(X)", "M", 1, "-M [1|2|3]"));

		result.addAll(Collections.list(super.listOptions()));

		return result.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 *
	 * <pre>
	 * -M [1|2|3]
	 *  The method used in transformation:
	 *  1.arithmatic average; 2.geometric centor;
	 *  3.using minimax combined features of a bag (default: 1)
	 *
	 *  Method 3:
	 *  Define s to be the vector of the coordinate-wise maxima
	 *  and minima of X, ie.,
	 *  s(X)=(minx1, ..., minxm, maxx1, ...,maxxm), transform
	 *  the exemplars into mono-instance which contains attributes
	 *  s(X)
	 * </pre>
	 *
	 * <pre>
	 * -W
	 *  Full name of base classifier.
	 *  (default: weka.classifiers.rules.ZeroR)
	 * </pre>
	 *
	 * <pre>
	 * Options specific to classifier weka.classifiers.rules.ZeroR:
	 * </pre>
	 *
	 * <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		String methodString = Utils.getOption('M', options);
		if (methodString.length() != 0) {
			setTransformMethod(new SelectedTag(Integer.parseInt(methodString),
					TAGS_TRANSFORMMETHOD));
		} else {
			setTransformMethod(new SelectedTag(TRANSFORMMETHOD_ARITHMETIC,
					TAGS_TRANSFORMMETHOD));
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		result.add("-M");
		result.add("" + m_TransformMethod);

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String transformMethodTipText() {
		return "The method used in transformation.";
	}

	/**
	 * Set the method used in transformation.
	 *
	 * @param newMethod the index of method to use.
	 */
	public void setTransformMethod(SelectedTag newMethod) {
		if (newMethod.getTags() == TAGS_TRANSFORMMETHOD) {
			m_TransformMethod = newMethod.getSelectedTag().getID();
		}
	}

	/**
	 * Get the method used in transformation.
	 *
	 * @return the index of method used.
	 */
	public SelectedTag getTransformMethod() {
		return new SelectedTag(m_TransformMethod, TAGS_TRANSFORMMETHOD);
	}

	/**
	 * Implements MITransform (3 type of transformation) 1.arithmatic average;
	 * 2.geometric centor; 3.merge minima and maxima attribute value together
	 *
	 * @param train the multi-instance dataset (with relational attribute)
	 * @return the transformed dataset with each bag contain mono-instance
	 *         (without relational attribute) so that any classifier not for MI
	 *         dataset can be applied on it.
	 * @throws Exception if the transformation fails
	 */
	public Instances transform(Instances train) throws Exception {

		Attribute classAttribute = (Attribute) train.classAttribute().copy();
		Attribute bagLabel = train.attribute(0);
		double labelValue;

		Instances newData = train.attribute(1).relation().stringFreeStructure();

		// insert a bag label attribute at the begining
		newData.insertAttributeAt(bagLabel, 0);

		// insert a class attribute at the end
		newData.insertAttributeAt(classAttribute, newData.numAttributes());
		newData.setClassIndex(newData.numAttributes() - 1);

		Instances mini_data = newData.stringFreeStructure();
		Instances max_data = newData.stringFreeStructure();

		Instance newInst = new DenseInstance(newData.numAttributes());
		Instance mini_Inst = new DenseInstance(mini_data.numAttributes());
		Instance max_Inst = new DenseInstance(max_data.numAttributes());
		newInst.setDataset(newData);
		mini_Inst.setDataset(mini_data);
		max_Inst.setDataset(max_data);

		double N = train.numInstances();// number of bags
		for (int i = 0; i < N; i++) {
			int attIdx = 1;
			Instance bag = train.instance(i); // retrieve the bag instance
			labelValue = bag.value(0);
			if (m_TransformMethod != TRANSFORMMETHOD_MINIMAX) {
				newInst.setValue(0, labelValue);
			} else {
				mini_Inst.setValue(0, labelValue);
				max_Inst.setValue(0, labelValue);
			}

			Instances data = bag.relationalValue(1); // retrieve relational value for
			// each bag
			for (int j = 0; j < data.numAttributes(); j++) {
				double value;
				if (m_TransformMethod == TRANSFORMMETHOD_ARITHMETIC) {
					value = data.meanOrMode(j);
					newInst.setValue(attIdx++, value);
				} else if (m_TransformMethod == TRANSFORMMETHOD_GEOMETRIC) {
					double[] minimax = minimax(data, j);
					value = (minimax[0] + minimax[1]) / 2.0;
					newInst.setValue(attIdx++, value);
				} else { // m_TransformMethod == TRANSFORMMETHOD_MINIMAX
					double[] minimax = minimax(data, j);
					mini_Inst.setValue(attIdx, minimax[0]);// minima value
					max_Inst.setValue(attIdx, minimax[1]);// maxima value
					attIdx++;
				}
			}

			if (m_TransformMethod == TRANSFORMMETHOD_MINIMAX) {
				if (!bag.classIsMissing()) {
					max_Inst.setClassValue(bag.classValue()); // set class value
				}
				mini_data.add(mini_Inst);
				max_data.add(max_Inst);
			} else {
				if (!bag.classIsMissing()) {
					newInst.setClassValue(bag.classValue()); // set class value
				}
				newData.add(newInst);
			}
		}

		if (m_TransformMethod == TRANSFORMMETHOD_MINIMAX) {
			// delete class attribute for the minima data
			mini_data.setClassIndex(-1);
			mini_data.deleteAttributeAt(mini_data.numAttributes() - 1);
			// delete the bag label attribute for the maxima data
			max_data.deleteAttributeAt(0);
			// merge minima and maxima data
			newData = mergeInstances(mini_data, max_data);
			newData.setClassIndex(newData.numAttributes() - 1);
		}

		return newData;
	}

	/**
	 * Get the minimal and maximal value of a certain attribute in a certain data
	 *
	 * @param data the data
	 * @param attIndex the index of the attribute
	 * @return the double array containing in entry 0 for min and 1 for max.
	 */
	public static double[] minimax(Instances data, int attIndex) {
		double[] rt = { Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };
		for (int i = 0; i < data.numInstances(); i++) {
			double val = data.instance(i).value(attIndex);
			if (val > rt[1]) {
				rt[1] = val;
			}
			if (val < rt[0]) {
				rt[0] = val;
			}
		}

		for (int j = 0; j < 2; j++) {
			if (Double.isInfinite(rt[j])) {
				rt[j] = Double.NaN;
			}
		}

		return rt;
	}

	public static Instances mergeInstances(Instances min, Instances max) {
		if (min.numInstances() != max.numInstances()) {
			throw new IllegalArgumentException("Instance sets must be of the same size");
		} else {
			ArrayList<Attribute> newAttributes = new ArrayList<>(min.numAttributes() + max.numAttributes());

			newAttributes.add(min.attribute(0));
			for(int i = 1; i < min.numAttributes(); ++i) {
				newAttributes.add(new Attribute("min" + min.attribute(i).name()));
			}
			for(int i = 0; i < max.numAttributes() - 1; ++i) {
				newAttributes.add(new Attribute("max" + max.attribute(i).name()));
			}
			newAttributes.add(max.attribute(max.numAttributes() - 1));

			Instances merged = new Instances(min.relationName() + '_' + max.relationName(), newAttributes, min.numInstances());

			for(int i = 0; i < min.numInstances(); ++i) {
				merged.add(min.instance(i).mergeInstance(max.instance(i)));
			}

			return merged;
		}
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.RELATIONAL_ATTRIBUTES);
		result.disable(Capability.MISSING_VALUES);

		// class
		result.disableAllClasses();
		result.disableAllClassDependencies();
		if (super.getCapabilities().handles(Capability.NOMINAL_CLASS)) {
			result.enable(Capability.NOMINAL_CLASS);
		}
		if (super.getCapabilities().handles(Capability.BINARY_CLASS)) {
			result.enable(Capability.BINARY_CLASS);
		}
		result.enable(Capability.MISSING_CLASS_VALUES);

		// other
		result.enable(Capability.ONLY_MULTIINSTANCE);

		return result;
	}

	/**
	 * Returns the capabilities of this multi-instance classifier for the
	 * relational data.
	 *
	 * @return the capabilities of this object
	 * @see Capabilities
	 */
	@Override
	public Capabilities getMultiInstanceCapabilities() {
		Capabilities result = super.getCapabilities();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.disableAllClasses();
		result.enable(Capability.NO_CLASS);

		return result;
	}

	/**
	 * Builds the classifier
	 *
	 * @param train the training data to be used for generating the boosted
	 *          classifier.
	 * @throws Exception if the classifier could not be built successfully
	 */
	@Override
	public void buildClassifier(Instances train) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(train);

		// remove instances with missing class
		train = new Instances(train);
		train.deleteWithMissingClass();

		if (m_Classifier == null) {
			throw new Exception("A base classifier has not been specified!");
		}

		if (getDebug()) {
			System.out.println("Start training ...");
		}
		Instances data = transform(train);

		data.deleteAttributeAt(0); // delete the bagID attribute
		m_Classifier.buildClassifier(data);

		if (getDebug()) {
			System.out.println("Finish building model");
		}
	}

	/**
	 * Computes the distribution for a given exemplar
	 *
	 * @param newBag the exemplar for which distribution is computed
	 * @return the distribution
	 * @throws Exception if the distribution can't be computed successfully
	 */
	@Override
	public double[] distributionForInstance(Instance newBag) throws Exception {

		double[] distribution = new double[2];
		Instances test = new Instances(newBag.dataset(), 0);
		test.add(newBag);

		test = transform(test);
		test.deleteAttributeAt(0);
		Instance newInst = test.firstInstance();

		distribution = m_Classifier.distributionForInstance(newInst);

		return distribution;
	}

	/**
	 * Gets a string describing the classifier.
	 *
	 * @return a string describing the classifer built.
	 */
	@Override
	public String toString() {
		return "SimpleMI with base classifier: \n" + m_Classifier.toString();
	}

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10369 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain the command line arguments to the scheme (see
	 *          Evaluation)
	 */
	public static void main(String[] argv) {
		runClassifier(new weka.classifiers.mi.SimpleMI(), argv);
	}
}
