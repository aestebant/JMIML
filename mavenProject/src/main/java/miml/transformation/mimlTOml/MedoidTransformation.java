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
package miml.transformation.mimlTOml;

import java.util.ArrayList;

import miml.clusterers.KMedoids;
import miml.core.distance.AverageHausdorff;
import miml.core.distance.IDistance;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class implementing the medoid-based transformation described in [1] to
 * transform an MIML problem to ML.
 * [1] <em> Zhou, Z. H., Zhang, M. L., Huang, S. J., &amp; Li, Y. F. (2012).
 * Multi-instance multi-label learning. Artificial Intelligence, 176(1),
 * 2291-2320. </em>
 * This class requires method transformDataset to have been executed before
 * executing transformInstance method.
 *
 * @author Eva Gibaja
 * @author Aurora Esteban
 * @version 20230412
 */
public class MedoidTransformation extends MIMLtoML {

    /**
     * For serialization
     */
    private static final long serialVersionUID = 94921720184805609L;

    /**
     * Clusterer.
     */
    private KMedoids kmedoids = null;

    private IDistance distanceMetric;

    /**
     * True if the resulting transformed dataset will be normalized to (0,1) with min-max normalization. By default,
     * False. If a learning algorithm that uses a NormalizableDistance is going to be used after transformation,
     * normalization is not needed.
     */
    private Boolean normalize;

    /**
     * If it is different to -1 this value represent that the number of clusters
     * will be a percentage of the number of instances of the dataset. For instance
     * 0.2 represents that the number of clusters is the 20% of the number of
     * instances, 0.45 a 45%, and so on. If this value is -1 the number of clusters
     * to consider is represented by numberOfClusters property. If the number of
     * clusters is not set neither by percentage nor by the numberOfClusters
     * property, it will be considered by default a 20% of the number of instances
     * in the dataset. If both the percentage and the numberOfClusters are set, the
     * percentage will be applied.
     */
    private float percentClusters = -1;

    /**
     * The number of clusters for kmedoids.
     */
    private int numClusters = -1;

    /**
     * Whether the clustering step has been executed or not. This class requires
     * method transformDataset to have been executed before executing
     * transformInstance method.
     */
    boolean clusteringDone = false;

    /**
     * Constructor.
     */
    public MedoidTransformation() throws Exception {
        super();
        this.percentClusters = 0.2F;
        this.normalize = false;
        this.distanceMetric = new AverageHausdorff();
    }

    /**
     * Constructor. Uses the same default number of clusters as MIMLSVM: 20% of
     * number of bags
     *
     * @param dataset MIMLInstances dataset.
     * @throws Exception To be handled in an upper level.
     */
    public MedoidTransformation(MIMLInstances dataset) throws Exception {
        this(0.2F, false, new AverageHausdorff());
        this.dataset = dataset;
    }

    /**
     * Constructor.
     *
     * @param dataset     MIMLInstances dataset.
     * @param numClusters number of clusters for k-medoids.
     * @throws Exception To be handled in an upper level.
     */
    public MedoidTransformation(MIMLInstances dataset, int numClusters) throws Exception {
        this(numClusters, false, new AverageHausdorff());
        this.dataset = dataset;
    }

    /**
     * Constructor.
     *
     * @param dataset    MIMLInstances dataset.
     * @param percentClusters The number of clusters for k-medoids as a percentage of the
     *                   number of bags. It is a value in (0,1). For instance, 0.2
     *                   is 20%.
     * @throws Exception To be handled in an upper level.
     */
    public MedoidTransformation(MIMLInstances dataset, float percentClusters) throws Exception {
        this(percentClusters, false, new AverageHausdorff());
        this.dataset = dataset;
    }

    public MedoidTransformation(float percentClusters, boolean normalize) throws Exception {
        this(percentClusters, normalize, new AverageHausdorff());
    }

    public MedoidTransformation(int numClusters, boolean normalize) throws Exception {
        this(numClusters, normalize, new AverageHausdorff());
    }

    /**
     * Constructor.
     *
     * @param percentClusters The number of clusters for k-medoids as a percentage of the number of bags. It is a value in (0,1). For instance, 0.2 is 20%.
     * @param distanceMetric      The distance function to be used by k-medoids.
     * @throws Exception To be handled in an upper level.
     */
    public MedoidTransformation(float percentClusters, boolean normalize, IDistance distanceMetric) throws Exception {
        this.percentClusters = percentClusters;
        this.normalize = normalize;
        this.distanceMetric = distanceMetric;
    }

    public MedoidTransformation(int numClusters, boolean normalize, IDistance distanceMetric) throws Exception {
        this.numClusters = numClusters;
        this.normalize = normalize;
        this.distanceMetric = distanceMetric;
    }

    protected void clusteringStep() throws Exception {
        if (this.numClusters != -1) {
            System.out.println("Number of clusters fixed: " + this.numClusters);
        } else {
            this.numClusters = (int) (this.dataset.getNumBags() * this.percentClusters);
            System.out.println("Number of clusters by percentaje (" + this.percentClusters + "): " + this.numClusters);
        }
        this.kmedoids = new KMedoids(this.numClusters, 1000, this.distanceMetric);

        System.out.println("Medoid Transformation.\n\tPerforming k-medoids clustering to transform the dataset.");
        kmedoids.buildClusterer(dataset.getDataSet());
        clusteringDone = true;
        System.out.println("\t" + kmedoids.numberOfClusters() + " clusters in " + kmedoids.getNumIterations() + " iterations");
        prepareTemplate();
        template.setRelationName(dataset.getDataSet().relationName() + "_medoid_transformation");
    }

    @Override
    public MultiLabelInstances transformDataset() throws Exception {
        // Clustering with kmedoids step
        clusteringStep();

        // Transformation step
        Instances newData = new Instances(template);
        int[] labelIndices = dataset.getLabelIndices();
        Instance newInst = new DenseInstance(newData.numAttributes());
        newInst.setDataset(newData); // Sets the reference to the dataset

        // For all bags in the dataset
        double nBags = dataset.getNumBags();
        int numClusters = kmedoids.numberOfClusters();
        for (int i = 0; i < nBags; i++) {

            // retrieves a bag
            MIMLBag bag = dataset.getBag(i);

            // sets the bagLabel
            newInst.setValue(0, bag.value(0));

            // computes distances to medoids
            double[] distance = kmedoids.distanceToMedoids(bag);

            // an attribute for medoid
            for (int k = 0, attIdx = 1; k < numClusters; k++, attIdx++) {
                newInst.setValue(attIdx, distance[k]);
            }
            // Copy label information into the dataset
            for (int j = 0; j < labelIndices.length; j++) {
                newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
            }
            newData.add(newInst);
        }
        if (normalize) {
            System.out.println("\t Performing min-max normalization on the transformed dataset.");
            return normalize(new MultiLabelInstances(newData, dataset.getLabelsMetaData()));
        } else
            return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
    }

    @Override
    public MultiLabelInstances transformDataset(MIMLInstances dataset) throws Exception {
        this.dataset = dataset;
        return transformDataset();
    }

    /**
     * Normalizes a multi-label dataset performing min-max normalization.
     *
     * @param dataset The dataset to be normalized.
     * @return Returns the normalized dataset as MultiLabelInstances.
     * @throws Exception To be handled in an upper level.
     */
    protected MultiLabelInstances normalize(MultiLabelInstances dataset) throws Exception {

        // 1. Computes statistics to perform normalization
        // number of attributes including the bagID attribute
        int nFeatures = dataset.getFeatureAttributes().size();
        double[] Max = new double[nFeatures];
        double[] Min = new double[nFeatures];
        double[] Range = new double[nFeatures];

        for (int i = 0; i < nFeatures; i++) {
            Max[i] = Double.NEGATIVE_INFINITY;
            Min[i] = Double.POSITIVE_INFINITY;
            Range[i] = 0;
        }

        boolean isNormalized = true;
        for (int i = 0; i < dataset.getNumInstances(); i++) {
            Instance instance = dataset.getDataSet().instance(i);
            // j=1 to ignore the bagId attribute
            for (int j = 1; j < nFeatures; j++) {
                if (instance.attribute(j).isNumeric()) {
                    if (instance.value(j) < 0 || instance.value(j) > 1)
                        isNormalized = false;
                    if (instance.value(j) < Min[j])
                        Min[j] = instance.value(j);
                    if (instance.value(j) > Max[j])
                        Max[j] = instance.value(j);
                }
            }
        }

        // j=1 to ignore the bagId attribute
        for (int i = 1; i < nFeatures; i++) {
            Range[i] = Max[i] - Min[i];
        }

        // 2. Normalizes the dataset
        if (isNormalized)
            return dataset;
        else {
            for (int i = 0; i < dataset.getNumInstances(); i++) {
                Instance instance = dataset.getDataSet().instance(i);

                // j=1 to ignore the bagId attribute
                for (int j = 1; j < nFeatures; j++) {
                    double value = instance.value(j);

                    // to avoid dividing by zero in case of a 0 range
                    if (Double.compare(Min[j], Max[j]) != 0) {
                        value = (value - Min[j]) / (Range[j]);
                    } else {
                        value = 1;
                    }
                    instance.setValue(j, value);
                }
            }
            return dataset;
        }
    }

    /**
     * Returns the value of the property normalize.
     *
     * @return The value of the property normalize.
     */
    public Boolean getNormalize() {
        return normalize;
    }

    /**
     * Sets the property normalized. If true, the resulting transformed multi-label
     * dataset will be normalized after transformation.
     *
     * @param normalize The value of the property to be set.
     */
    public void setNormalize(Boolean normalize) {
        this.normalize = normalize;
    }

    @Override
    public Instance transformInstance(MIMLBag bag) throws Exception {
        if (!clusteringDone)
            throw new Exception(
                    "The transformInstance method must be called after executing transformDataset that performs kmedoids clustering required by this kind of transformation.");

        int[] labelIndices = dataset.getLabelIndices();
        Instance newInst = new DenseInstance(template.numAttributes());

        // sets the bagLabel
        newInst.setDataset(bag.dataset()); // Sets the reference to the dataset
        newInst.setValue(0, bag.value(0));

        // computes distances to medoids
        double[] distance = this.kmedoids.distanceToMedoids(bag);

        // an attribute for medoid
        int numClusters = kmedoids.numberOfClusters();
        for (int k = 0, attIdx = 1; k < numClusters; k++, attIdx++) {
            newInst.setValue(attIdx, distance[k]);
        }

        // Insert label information into the instance
        for (int j = 0; j < labelIndices.length; j++) {
            newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
        }
        return newInst;
    }

    @Override
    protected void prepareTemplate() throws Exception {
        int attrIndex = 0;
        ArrayList<Attribute> attributes = new ArrayList<>();

        // insert a bag label attribute at the beginning
        Attribute attr = dataset.getDataSet().attribute(0);
        attributes.add(attr);

        // Adds attributes for medodis
        int numClusters = kmedoids.numberOfClusters();
        for (int k = 1; k <= numClusters; k++) {
            attr = new Attribute("distanceToMedoid_" + k);
            attributes.add(attr);
            attrIndex++;
        }

        // Insert labels as attributes in the dataset
        int[] labelIndices = dataset.getLabelIndices();
        updatedLabelIndices = new int[labelIndices.length];
        ArrayList<String> values = new ArrayList<>(2);
        values.add("0");
        values.add("1");
        for (int i = 0; i < labelIndices.length; i++) {
            attr = new Attribute(dataset.getDataSet().attribute(labelIndices[i]).name(), values);
            attributes.add(attr);
            attrIndex++;
            updatedLabelIndices[i] = attrIndex;
        }

        template = new Instances("templateMedoid", attributes, 0);
    }

    /**
     * Returns the number of clusters.
     *
     * @return Returns the number of clusters to perform clustering.
     * @throws Exception To be handled in an upper level.
     */
    public int numberOfClusters() throws Exception {
        return numClusters;
    }

    /**
     * Sets the number of clusters to perform clustering. This method must be called
     * before clustering.
     *
     * @param numClusters A number of clusters.
     */
    public void setNumClusters(int numClusters) {
        this.numClusters = numClusters;
    }

    public float getPercentClusters() {
        return percentClusters;
    }

    public void setPercentClusters(float percentClusters) {
        this.percentClusters = percentClusters;
    }

    /**
     * Gets the maximum number of iterations used by clusterer.
     *
     * @return The maximum number of iterations.
     */
    public int getMaxIterations() {
        return this.kmedoids.getMaxIterations();
    }

    /**
     * Sets the maximum number of iterations for clustering. This method must be
     * called before clustering.
     *
     * @param maxIterations The maximum number of iterations for clustering.
     */
    public void setMaxIterations(int maxIterations) {
        this.kmedoids.setMaxIterations(maxIterations);
    }

    /**
     * Returns the distance function used for clustering.
     *
     * @return The distance function used for clustering.
     */
    public IDistance getDistanceFunction() {
        return this.distanceMetric;
    }

    /**
     * Sets the distance function to use for clustering. This method must be called
     * before clustering.
     *
     * @param distanceFunction The distance function used for clustering.
     */
    public void setDistanceFunction(IDistance distanceFunction) {
        this.distanceMetric = distanceFunction;
    }
}
