package miml.transformation.mimlTOml;
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

import java.util.ArrayList;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.clusterers.SimpleKMeans;
import weka.core.*;

/**
 * Class implementing the kmeans-based transformation described in [1] to
 * transform an MIML problem to ML.
 * <p>
 * [1] <em>Li, Y. F., Hu, J. H., Jiang, Y., and Zhou, Z. H. (2012). Towards
 * discovering what patterns trigger what labels. In Proceedings of the AAAI
 * Conference on Artificial Intelligence (Vol. 26, No. 1, pp. 1012-1018).</em>
 * <p>
 * This class requires method transformDataset to have been executed before
 * executing transformInstance method.
 *
 * @author Eva Gibaja
 * @version 20231115
 */

public class KMeansTransformation extends MIMLtoML {

    /** For serialization. */
    private static final long serialVersionUID = 1L;

    /** Clusterer. */
    protected SimpleKMeans clusterer = null;

    /**
     * If it is different to -1 this value represent that the number of clusters
     * will be a percentage of the number of training bags in the dataset. For
     * instance 0.2 represents that the number of clusters is the 20% of the number
     * of training bags, 0.45 a 45%, and so on. If this value is -1 the number of
     * clusters to consider is represented by numClusters property. If the number of
     * clusters is not set neither by percentage nor by the numClusters property, it
     * will be considered by default a 50% of the number of training bags in the
     * dataset. If both the percentage and the numClusters are set, the percentage
     * will be applied.
     */
    protected double percentClusters = -1;

    /** The number of clusters. */
    protected int numClusters = -1;

    /** The seed for kmeans clustering. By default, 1. */
    protected int seed = 1;

    /** Whether the clustering step has been executed or not. */
    protected boolean clusteringDone = false;

    /**
     * Clustering prototypes obtained each one as the nearest instance to each
     * centroid.
     */
    protected Instances prototypes;

    /**
     * The delta value for each cluster obtained as the average distance between
     * instances in each cluster
     */
    protected double[] delta;

    protected DistanceFunction dfunc = null;

    public KMeansTransformation(float percentClusters, int seed) throws Exception {
        this.percentClusters = percentClusters;
        this.seed = seed;
    }
    public KMeansTransformation(int numClusters, int seed) throws Exception {
        this.numClusters = numClusters;
        this.seed = seed;
    }

    public KMeansTransformation(float percentClusters) throws Exception {
        this(percentClusters, 1);
    }
    public KMeansTransformation(int numClusters) throws Exception {
        this(numClusters, 1);
    }

    public KMeansTransformation() throws Exception {
        this(0.5F, 1);
    }

    @Override
    public MultiLabelInstances transformDataset() throws Exception {

        // 1. CLUSTERING STEP
        // prepares the single-instance dataset for clustering
        double nBags = dataset.getNumBags();
        Instances singleInstances = new Instances(dataset.getBag(0).getBagAsInstances());
        for (int i = 1; i < nBags; i++) {
            singleInstances.addAll(dataset.getBag(i).getBagAsInstances());
        }

        this.configureClusterer();

        // performs clustering
        System.out.println("k-means Transformation.\n\tPerforming k-means clustering to transform the dataset");
        clusterer.buildClusterer(singleInstances);
        clusteringDone = true;
        System.out.println("\tnClusters=" + clusterer.getNumClusters());
        System.out.println("\tseed=" + clusterer.getSeed());
        prepareTemplate();
        template.setRelationName(dataset.getDataSet().relationName() + "_kmeans_transformation");
        dfunc = new EuclideanDistance();
        dfunc.setInstances(singleInstances);
        // prototypes are the instances closest to the cluster centroids
        Instances centroids = clusterer.getClusterCentroids();
        for (int k = 0; k < centroids.numInstances(); k++)
            dfunc.update(centroids.instance(k));
        double[][] distanceMatrix = computeDistanceMatrix(centroids, singleInstances);
        int[] clusterAssignment = clusterAssignment(distanceMatrix);
        int[] prototypesIndex = computeIndexPrototypes(distanceMatrix);
        int nClusters = distanceMatrix[0].length;
        prototypes = new Instances(singleInstances, 0); // creates empty datasest
        for (int k = 0; k < nClusters; k++) prototypes.add(singleInstances.instance(prototypesIndex[k]));
        // computes the value of delta as the average distance between instances in one cluster
        double[] delta = computeDelta(clusterAssignment, singleInstances);

        // 2. TRANSFORMATION STEP
        Instances newData = new Instances(template);
        int[] labelIndices = dataset.getLabelIndices();
        Instance newInst = new DenseInstance(newData.numAttributes());
        newInst.setDataset(newData); // Sets the reference to the dataset
        for (int i = 0; i < nBags; i++) {
            MIMLBag bag = dataset.getBag(i);
            // sets the bagLabel
            newInst.setValue(0, bag.value(0));
            for (int k = 0, attIdx = 1; k < nClusters; k++, attIdx++) {
                double sim = similarity(singleInstances.instance(prototypesIndex[k]), bag, delta[k]);
                newInst.setValue(attIdx, sim);
            }
            // Copy label information into the dataset
            for (int j = 0; j < labelIndices.length; j++) {
                newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
            }
            newData.add(newInst);
        }
        return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
    }

    @Override
    public MultiLabelInstances transformDataset(MIMLInstances dataset) throws Exception {
        if (!clusteringDone) {
            this.dataset = dataset;
            return transformDataset();
        }
        Instances newData = new Instances(this.template);
        for (int i = 0; i < dataset.getNumBags(); ++i) {
            MIMLBag bag = dataset.getBag(i);
            Instance transformedBag = this.transformInstance(bag);
            newData.add(transformedBag);
        }
        return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
    }

    @Override
    public Instance transformInstance(MIMLBag bag) throws Exception {
        if (!clusteringDone)
            throw new Exception("The transformInstance method must be called after executing transformDataset that performs kmeans clustering required by this kind of transformation.");

        int[] labelIndices = dataset.getLabelIndices();
        Instance newInst = new DenseInstance(template.numAttributes());
        // sets the bagLabel
        newInst.setDataset(bag.dataset()); // Sets the reference to the dataset
        newInst.setValue(0, bag.value(0));
        // an attribute per centroid
        int numClusters = clusterer.getNumClusters();
        for (int k = 0, attIdx = 1; k < numClusters; k++, attIdx++) {
            double sim = similarity(prototypes.instance(k), bag, delta[k]);
            newInst.setValue(attIdx, sim);
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
        // Adds attributes for prototypes
        int numClusters = clusterer.getNumClusters();
        for (int k = 1; k <= numClusters; k++) {
            attr = new Attribute("similarityToPrototype_" + k);
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
        template = new Instances("templatePrototype", attributes, 0);
    }

    /**
     * Computes the distance matrix between centroids and single instances used for
     * clustering.
     *
     * @param centroids       The centroids obtained by kmeans clustering.
     * @param singleInstances The single-instance instances used for clustering.
     * @return A matrix of double in which matrix[i][k] stores the distance from
     *         instance i to centroid k.
     */
    protected double[][] computeDistanceMatrix(Instances centroids, Instances singleInstances) {

        int nInstances = singleInstances.numInstances();
        int nClusters = centroids.numInstances();
        double[][] distanceMatrix = new double[nInstances][nClusters];

        for (int i = 0; i < nInstances; i++) {
            for (int k = 0; k < nClusters; k++) {
                double dist = dfunc.distance(centroids.instance(k), singleInstances.instance(i));
                distanceMatrix[i][k] = dist;
            }
        }
        return distanceMatrix;
    }

    /**
     * Computes a vector of nClusters elements with the index of the prototypes
     * obtained as the closest instance to each centroid.
     *
     * @param distanceMatrix A matrix of nInstancesxnClusters with the distance
     *                       between centroids and single-instances used for
     *                       clustering . Matrix[i][k] is the distance from instance
     *                       i to centroid k.
     * @return A vector with the index of the prototypes in the dataset of
     *         single-instances used for clustering. This vector is obtained as the
     *         index of the minimum row of each column.
     * @throws Exception To be handled in an upper level.
     */
    protected int[] computeIndexPrototypes(double[][] distanceMatrix) throws Exception {

        int nInstances = distanceMatrix.length;
        int nClusters = distanceMatrix[0].length;
        int[] minIndex = new int[nClusters];

        for (int k = 0; k < nClusters; k++) {
            minIndex[k] = 0;
            for (int i = 1; i < nInstances; i++) {
                if (distanceMatrix[i][k] < distanceMatrix[minIndex[k]][k]) {
                    minIndex[k] = i;
                }
            }
        }
        return minIndex;
    }

    /**
     * Computes a vector of nInstances with the index of the cluster assigned to
     * each instance.
     *
     * @param distanceMatrix A matrix of nInstancesxnClusters with the distance
     *                       between centroids and single-instances used for
     *                       clustering . Matrix[i][k] is the distance from instance
     *                       i to centroid k.
     * @return A vector with the index of the cluster assigned to each instance.
     *         This vector is obtained as the index of the minimum column of each
     *         row.
     */
    protected int[] clusterAssignment(double[][] distanceMatrix) {

        int nInstances = distanceMatrix.length;
        int nClusters = distanceMatrix[0].length;
        int[] assignment = new int[nInstances];

        for (int i = 0; i < nInstances; i++) {
            assignment[i] = 0;
            for (int k = 1; k < nClusters; k++) {
                if (distanceMatrix[i][k] < distanceMatrix[i][assignment[i]])
                    assignment[i] = k;
            }
        }
        return assignment;
    }

    /**
     * Computes similarity between a centroid, represented by a single instance, and
     * a bag. The value is computed as Gaussian distance.
     *
     * @param centroid A centroid.
     * @param bag      A bag.
     * @param delta_k  A vector with a delta value for each centroid.
     * @return The similarity, a value normalized to [0,1].
     * @throws Exception To be handled in an upper level.
     */
    protected double similarity(Instance centroid, MIMLBag bag, double delta_k) throws Exception {
        double min_sim = 0;
        Instances instances = bag.getBagAsInstances();
        for (int j = 0; j < instances.numInstances(); j++) {
            double dist = dfunc.distance(centroid, instances.instance(j));
            double sim = Math.exp(-((dist * dist) / delta_k));
            if (j == 0) min_sim = sim;
            if (sim < min_sim) min_sim = sim;
        }
        return min_sim;
    }

    /**
     * Computes the delta value for each cluster that is used for similarity
     * computation. This value is computed as the average distance between all pair
     * of instances in each cluster.
     *
     * @param clusterAssignment A vector of nInstances elements with the indices of
     *                          the clusters assigned to each one.
     * @param singleInstances   The instances used for clustering.
     * @return A vector of nClusters with the delta value for each cluster.
     */
    protected double[] computeDelta(int[] clusterAssignment, Instances singleInstances) {
        int nClusters = clusterAssignment.length;
        int nInstances = singleInstances.numInstances();
        delta = new double[nClusters];

        for (int k = 0; k < nClusters; k++) {
            double sumDistances_k = 0;
            int instances_k = 0;
            for (int i = 0; i < nInstances; i++) {
                if (clusterAssignment[i] == k) {
                    for (int j = i + 1; j < nInstances; j++) {
                        if (clusterAssignment[j] == k) {
                            instances_k++;
                            sumDistances_k = dfunc.distance(singleInstances.instance(i), singleInstances.instance(j));
                        }
                    }

                }
            }
            delta[k] = sumDistances_k / instances_k;
        }
        return delta;
    }

    /**
     * Determines the number of cluster depending on the values of the properties
     * percentage and numClusters. Sets the number of clusters and the seed for
     * clustering.
     *
     * @throws Exception To be handled in an upper level.
     */
    void configureClusterer() throws Exception {
        if (this.numClusters == -1) {
            this.numClusters = (int) (this.dataset.getNumBags() * this.percentClusters);
        }
        if (this.numClusters > dataset.getNumBags()) {
            System.out.println("WARNING: more clusters than bags configured, changing numClusters to " + dataset.getNumBags());
            this.numClusters = dataset.getNumBags();
        }
        if (this.numClusters < 2) {
            System.out.println("WARNING: less than 2 clusters configured, changing numClusters to 2");
            this.numClusters = 2;
        }
        this.clusterer = new SimpleKMeans();
        clusterer.setNumClusters(this.numClusters);
        clusterer.setSeed(this.seed);
        clusterer.setMaxIterations(100);
        clusterer.setDistanceFunction(new EuclideanDistance());
    }

    // --------------------
    // GETTERS AND SETTERS
    // --------------------

    /**
     * Returns the value of the seed of the clusterer.
     *
     * @return int
     */
    public int getSeed() {
        return seed;
    }

    /**
     * Sets the value of the seed used for clustering in both the transformer and the clusterer.
     *
     * @param seed The seed
     */
    public void setSeed(int seed) {
        this.seed = seed;
    }

    /**
     * Returns the number of clusters.
     *
     * @return Returns the number of clusters to perform clustering.
     * @throws Exception To be handled in an upper level.
     */
    public int getNumClusters() throws Exception {
        return numClusters;
    }

    /**
     * Sets the number of clusters to perform clustering in both the transformer and the clusterer.
     *
     * @param numClusters A number of clusters.
     */
    public void setNumClusters(int numClusters) {
        this.numClusters = numClusters;
    }

    public double getPercentClusters() {
        return percentClusters;
    }

    public void setPercentClusters(double percentClusters) {
        this.percentClusters = percentClusters;
    }
}