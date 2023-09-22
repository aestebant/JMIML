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

package miml.classifiers.miml.mimlTOml;

import miml.classifiers.miml.MIMLClassifier;
import miml.core.ConfigParameters;
import miml.core.Params;
import miml.core.Utils;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.transformation.mimlTOml.MIMLtoML;
import miml.transformation.mimlTOml.MedoidTransformation;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import org.apache.commons.configuration2.Configuration;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.lang.reflect.InvocationTargetException;
import java.util.Objects;

/**
 * <p>
 * Class implementing the transformation algorithm for MIML data to solve it
 * with ML learning. For more information, see <em>Zhou, Z. H., &#38; Zhang, M.
 * L. (2007). Multi-instance multi-label learning with application to scene
 * classification. In Advances in neural information processing systems (pp.
 * 1609-1616).</em>
 * </p>
 *
 * @author Aurora Esteban
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180608
 */
public class MIMLClassifierToML extends MIMLClassifier {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * A Generic MultiLabel classifier.
	 */
	protected MultiLabelLearner baseClassifier;

	/**
	 * The transform method.
	 */
	protected MIMLtoML transformationMethod;

	/**
	 * The miml dataset.
	 */
	protected MIMLInstances mimlDataset;

	Remove removeFilter;
	MultiLabelInstances mlDataSetWithBagId;

	/**
	 * Basic constructor to initialize the classifier.
	 *
	 * @param baseClassifier       The base classification algorithm.
	 * @param transformationMethod Algorithm used as transformation method from MIML
	 *                             to ML.
	 * @throws Exception To be handled in an upper level.
	 */
	public MIMLClassifierToML(MultiLabelLearner baseClassifier, MIMLtoML transformationMethod) throws Exception {
		super();
		this.baseClassifier = baseClassifier;
		this.transformationMethod = transformationMethod;
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public MIMLClassifierToML() {
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#buildInternal(data.MIMLInstances)
	 */
	@Override
	public void buildInternal(MIMLInstances mimlDataSet) throws Exception {

		this.mimlDataset = mimlDataSet;

		// Transforms a dataset
		mlDataSetWithBagId = transformationMethod.transformDataset(mimlDataSet);

		// Deletes bagIdAttribute
		removeFilter = new Remove();
		int[] indexToRemove = { 0 };
		removeFilter.setAttributeIndicesArray(indexToRemove);
		removeFilter.setInputFormat(mlDataSetWithBagId.getDataSet());
		Instances newData = Filter.useFilter(mlDataSetWithBagId.getDataSet(), removeFilter);

		MultiLabelInstances withoutBagId = new MultiLabelInstances(newData, mimlDataSet.getLabelsMetaData());

		baseClassifier.build(withoutBagId);
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#makePredictionInternal(data.Bag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag bag) throws Exception {

		Instance instance = transformationMethod.transformInstance(bag);

		// Delete bagIdAttribute
		Instances newData = new Instances(this.mlDataSetWithBagId.getDataSet(), 0);
		newData.add(instance);
		newData = Filter.useFilter(newData, removeFilter);

		return baseClassifier.makePrediction(newData.get(0));
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {
		Params params = Utils.readMultiLabelLearnerParams(configuration.subset("multiLabelClassifier"));
		this.configure(configuration, params);
	}

	public void configure(Configuration configuration, Params params) {
		// Get the string with the base classifier class
		String classifierName = configuration.getString("multiLabelClassifier[@name]");
		// Instance class
		Class<? extends MultiLabelLearner> classifierClass;
		try {
			classifierClass = Class.forName(classifierName).asSubclass(MultiLabelLearner.class);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
		try {
			this.baseClassifier = Objects.requireNonNull(classifierClass).getConstructor(params.getClasses())
					.newInstance(params.getObjects());
		} catch (InstantiationException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
			throw new RuntimeException(e);
		}
		// Get the string with the base classifier class
		String transformerName = configuration.getString("transformationMethod[@name]");
		// Instance class
		Class<? extends MIMLtoML> transformerClass;
		try {
			transformerClass = Class.forName(transformerName).asSubclass(MIMLtoML.class);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
		if (transformerClass == MedoidTransformation.class) {
			Configuration transformerConf = configuration.subset("transformationMethod");
			float k = transformerConf.getFloat("k");
			//TODO: add rest of possible constructors
			try {
				if (k < 1) {
					this.transformationMethod = Objects.requireNonNull(transformerClass).getConstructor(float.class).newInstance(k);
				} else {
					this.transformationMethod = Objects.requireNonNull(transformerClass).getConstructor(int.class).newInstance((int) k);
				}
			} catch (InstantiationException | IllegalAccessException | InvocationTargetException |
                     NoSuchMethodException e) {
				throw new RuntimeException(e);
			}
        } else {
			try {
				this.transformationMethod = Objects.requireNonNull(transformerClass).getConstructor().newInstance();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}

		ConfigParameters.setClassifierName(classifierName);
		ConfigParameters.setTransformationMethod(transformerName);
		ConfigParameters.setIsTransformation(true);
	}

}
