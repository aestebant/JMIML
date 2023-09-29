//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package miml.clusterers;

import java.util.Arrays;
import java.util.Random;
import miml.core.distance.AverageHausdorff;
import miml.core.distance.IDistance;
import weka.clusterers.Clusterer;
import weka.clusterers.RandomizableClusterer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class KMedoids extends RandomizableClusterer implements Clusterer {
    private static final long serialVersionUID = -6814942755920034118L;
    protected IDistance metric;
    protected int numClusters;
    protected int numInstances;
    protected int maxIterations;
    protected int[] medoidIndices;
    protected Instance[] medoidInstances;
    protected int[] clusterAssignment;
    protected double[][] distances;
    protected boolean minimize;
    protected boolean randomInitialization;
    protected double configurationCost;
    protected double numIterations;

    public KMedoids() throws Exception {
        this(10, 1000, new AverageHausdorff());
    }

    public KMedoids(int numClusters) throws Exception {
        this(numClusters, 1000, new AverageHausdorff());
    }

    public KMedoids(IDistance metric) throws Exception {
        this(10, 1000, metric);
    }

    public KMedoids(int numClusters, IDistance metric) throws Exception {
        this(numClusters, 1000, metric);
    }

    public KMedoids(int numClusters, int maxIterations, IDistance metric) throws Exception {
        this.numClusters = numClusters;
        this.maxIterations = maxIterations;
        this.minimize = true;
        this.randomInitialization = true;
        this.metric = metric;
    }

    protected boolean compare(double metricValue1, double metricValue2) {
        if (this.minimize) {
            return metricValue1 <= metricValue2;
        } else {
            return metricValue1 >= metricValue2;
        }
    }

    protected void randomInitialization() {
        Random rg = new Random(this.getSeed());

        for(int k = 0; k < this.numClusters; ++k) {
            int random = rg.nextInt(this.numInstances);
            while(this.isMedoid((random))) {
                random = rg.nextInt(this.numInstances);
            }
            this.medoidIndices[k] = random;
        }

    }

    protected void buildInitialization() {
        double[] sumDistances = new double[this.numInstances];
        int bestIndex = 0;
        for(int k = 0; k < this.numInstances; ++k) {
            sumDistances[k] = 0.0;

            for(int j = 0; j < this.numInstances; ++j) {
                if (j != k) {
                    sumDistances[k] += this.distances[k][j];
                }
            }

            if (this.compare(sumDistances[k], sumDistances[bestIndex])) {
                bestIndex = k;
            }
        }

        int k = 0;
        this.medoidIndices[k] = bestIndex;

        for(k = 1; k < this.numClusters; ++k) {
            double[] gain = new double[this.numInstances];
            Arrays.fill(gain, 0.0);

            int i;
            for(i = 0; i < this.numInstances; ++i) {
                if (!this.isMedoid(i)) {
                    for(int j = 0; j < this.numInstances; ++j) {
                        if (!this.isMedoid(j) && j != i) {
                            double Dj = this.distances[j][this.medoidIndices[0]];

                            for(int c = 1; c < k; ++c) {
                                if (this.compare(Dj, this.distances[j][c])) {
                                    Dj = this.distances[j][c];
                                }
                            }

                            if (!this.compare(Dj, this.distances[i][j])) {
                                gain[i] += Math.abs(Dj - this.distances[i][j]);
                            }
                        }
                    }
                }
            }

            bestIndex = 0;

            for(i = 1; i < this.numInstances; ++i) {
                if (!this.isMedoid(i) && gain[i] > gain[bestIndex]) {
                    bestIndex = i;
                }
            }

            this.medoidIndices[k] = bestIndex;
        }

    }

    public void buildClusterer(Instances data) throws Exception {
        this.numInstances = data.numInstances();
        if (this.numClusters > this.numInstances) {
            System.out.println("The number of clusters must be less or equal to the number of bags. Setting nClusters=" + this.numInstances);
            this.numClusters = this.numInstances;
        }

        if (this.numClusters < 2) {
            System.out.println("\nThe number of clusters must be at least 2. Setting numClusters=2");
            this.numClusters = 2;
        }

        this.metric.setInstances(data);
        this.distances = new double[this.numInstances][this.numInstances];
        this.computeDistances(data);
        this.medoidIndices = new int[this.numClusters];

        for(int k = 0; k < this.numClusters; ++k) {
            this.medoidIndices[k] = -1;
        }

        if (this.randomInitialization) {
            this.randomInitialization();
        } else {
            this.buildInitialization();
        }

        this.clusterAssignment = this.assignInstancesToMedoids(this.medoidIndices);
        double cost = this.computeCost(this.clusterAssignment);
        boolean change = true;

        int count;
        int k;
        for(count = 0; change && count < this.maxIterations; ++count) {
            change = false;

            for(k = 0; k < this.medoidIndices.length; ++k) {
                int oldMedoid = this.medoidIndices[k];

                for(int i = 0; i < data.numInstances(); ++i) {
                    if (!this.isMedoid(i)) {
                        this.medoidIndices[k] = i;
                        int[] candidateAsignment = this.assignInstancesToMedoids(this.medoidIndices);
                        double candidateCost = this.computeCost(candidateAsignment);
                        if (this.compare(candidateCost, cost)) {
                            this.clusterAssignment = candidateAsignment.clone();
                            cost = candidateCost;
                            change = true;
                        } else {
                            this.medoidIndices[k] = oldMedoid;
                        }
                    }
                }
            }
        }

        this.medoidInstances = new Instance[this.numClusters];

        for(k = 0; k < this.numClusters; ++k) {
            this.medoidInstances[k] = data.instance(this.medoidIndices[k]);
        }

        this.configurationCost = cost;
        this.numIterations = count;
    }

    protected void computeDistances(Instances data) throws Exception {
        for(int i = 0; i < data.numInstances(); ++i) {
            this.distances[i][i] = 0.0;
            Instance instanceA = data.instance(i);

            for(int j = i + 1; j < data.numInstances(); ++j) {
                Instance instanceB = data.instance(j);
                double dist = this.metric.distance(instanceA, instanceB);
                this.distances[i][j] = dist;
                this.distances[j][i] = dist;
            }
        }

    }

    protected int[] assignInstancesToMedoids(int[] medoidIndices) {
        this.clusterAssignment = new int[this.numInstances];

        for(int i = 0; i < this.numInstances; ++i) {
            int index = this.medoidIndex(i);
            if (index >= 0) {
                this.clusterAssignment[i] = index;
            } else {
                double bestDistance = this.distances[i][medoidIndices[0]];
                int bestMedoidIndex = 0;

                for(int k = 1; k < medoidIndices.length; ++k) {
                    double auxDistance = this.distances[i][medoidIndices[k]];
                    if (this.compare(auxDistance, bestDistance)) {
                        bestDistance = auxDistance;
                        bestMedoidIndex = k;
                    }
                }

                this.clusterAssignment[i] = bestMedoidIndex;
            }
        }

        return this.clusterAssignment;
    }

    protected double computeCost(int[] assignment) {
        double cost = 0.0;

        for(int i = 0; i < assignment.length; ++i) {
            cost += this.distances[i][this.medoidIndices[assignment[i]]];
        }

        return cost;
    }

    protected boolean isMedoid(int instanceIndex) {
        for (int medoidIndex : this.medoidIndices) {
            if (medoidIndex == instanceIndex) {
                return true;
            }
        }
        return false;
    }

    protected int medoidIndex(int instanceIndex) {
        for(int k = 0; k < this.medoidIndices.length; ++k) {
            if (this.medoidIndices[k] == instanceIndex) {
                return k;
            }
        }

        return -1;
    }

    public double[] distanceToMedoids(Instance instance) throws Exception {
        this.metric.update(instance);
        double[] distances = new double[this.numClusters];
        for(int k = 0; k < this.numClusters; ++k) {
            distances[k] = this.metric.distance(this.medoidInstances[k], instance);
        }
        return distances;
    }

    public double[] distanceToMedoids(int index) throws Exception {
        if (index < 0 || index > this.numInstances) {
            throw new Exception("Non registered instance with index " + index);
        }
        double[] distances = new double[this.numClusters];
        for(int k = 0; k < this.numClusters; ++k) {
            distances[k] = this.distances[index][this.medoidIndices[k]];
        }
        return distances;
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distances = this.distanceToMedoids(instance);
        double sumMinimize = 0.0;
        double sumMaximize = 0.0;

        for(int k = 0; k < this.numClusters; ++k) {
            sumMinimize += 1.0 / distances[k];
            sumMaximize += distances[k];
        }

        double[] distribution = new double[this.numClusters];

        for(int k = 0; k < this.numClusters; ++k) {
            if (this.minimize) {
                distribution[k] = 1.0 / distances[k] / sumMinimize;
            } else {
                distribution[k] = distances[k] / sumMaximize;
            }
        }

        return distribution;
    }

    public int clusterInstance(Instance instance) throws Exception {
        double[] evaluation = this.distributionForInstance(instance);
        return Utils.maxIndex(evaluation);
    }

    public int numberOfClusters() throws Exception {
        return this.numClusters;
    }

    public Capabilities getCapabilities() {
        return null;
    }

    public void setSeed(int seed) {
        super.setSeed(seed);
    }

    public int getSeed() {
        return super.getSeed();
    }

    public Instance[] getMedoidInstances() {
        return this.medoidInstances;
    }

    public IDistance getDistanceFunction() {
        return this.metric;
    }

    public void setDistanceFunction(IDistance distanceFunction) {
        this.metric = distanceFunction;
    }

    public void setNumberOfClusters(int numberOfClusters) {
        this.numClusters = numberOfClusters;
    }

    public int getMaxIterations() {
        return this.maxIterations;
    }

    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public boolean getRandomInitialization() {
        return this.randomInitialization;
    }

    public void setRandomInitialization(boolean randomInitialization) {
        this.randomInitialization = randomInitialization;
    }

    public int[] getAssignment() {
        return this.clusterAssignment;
    }

    public double getConfigurationCost() {
        return this.configurationCost;
    }

    public double getNumIterations() {
        return this.numIterations;
    }
}
