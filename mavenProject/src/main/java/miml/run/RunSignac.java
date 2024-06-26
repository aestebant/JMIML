package miml.run;

import miml.classifiers.miml.IMIMLClassifier;
import miml.core.ConfigLoader;
import miml.core.Params;
import miml.core.Utils;
import miml.evaluation.IEvaluator;
import miml.report.ExperimentReport;
import org.apache.commons.cli.*;
import org.apache.commons.configuration2.Configuration;
import org.apache.commons.configuration2.ex.ConfigurationException;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

public class RunSignac {
	public static void main(String[] args){
		// -l mi -t <br/lp/classifier> -c <base classifier> -o <options>
		// -l ml -c <base classifier> -t <arithmetic/geometric/minmax/medoid/means> -o <options>
		// -l ml -c <base classifier> -t medoid -o <options> -k <(0,1)/[1,n]>
		// -l miml -c <classifier> -o <options>
		// -a <train data> -e <test data> -x <xml file> -r <result file>
		Option learning = new Option("l", true, "learning: mi/ml/miml");
		learning.setRequired(true);
		Option transformation = new Option("t", true, "transformation: if MI=<br/lp/classifier>, if ML=<arithmetic/geometric/minmax/medoid/means>");
		Option nClusters = new Option("k", true, "K for the medoid/means transformation. If 0 < k < 1 -> percentage, else fixed number of clusters");
		Option classifier = new Option("c", true, "Base classifier");
		classifier.setRequired(true);
		Option classifierConfs = new Option("o", true, "Configurations for the base classifier");
		Option trainData = new Option("a", true, "Path to train data");
		trainData.setRequired(true);
		Option testData = new Option("e", true, "Path to test data");
		testData.setRequired(true);
		Option xmlData = new Option("x", true, "Path to XML data");
		xmlData.setRequired(true);
		Option report = new Option("r", true, "Path to report file");
		report.setRequired(true);

		Options options = new Options();
		options.addOption(learning);
		options.addOption(transformation);
		options.addOption(nClusters);
		options.addOption(classifier);
		options.addOption(classifierConfs);
		options.addOption(trainData);
		options.addOption(testData);
		options.addOption(xmlData);
		options.addOption(report);

		CommandLineParser parser = new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			formatter.printHelp("utility-name", options);
			System.exit(1);
		}

		ConfigLoader loader;
		try {
			loader = new ConfigLoader("configurations/base_config.xml");
		} catch (ConfigurationException e) {
			throw new RuntimeException(e);
		}

		Configuration configuration = loader.getConfiguration();
		Configuration mlParams = null;
		if (cmd.getOptionValue("l").equals("mi")) {
			configuration.setProperty("classifier[@name]", "miml.classifiers.miml.mimlTOmi.MIMLClassifierToMI");
			if (cmd.getOptionValue("t").equals("br"))
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.classifiers.miml.mimlTOmi.MIMLBinaryRelevance");
			else if (cmd.getOptionValue("t").equals("lp"))
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.classifiers.miml.mimlTOmi.MIMLLabelPowerset");
			else if (cmd.getOptionValue("t").equals("classifier"))
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.classifiers.miml.mimlTOmi.MIMLClassifierToMI");
			configuration.setProperty("classifier.multiInstanceClassifier[@name]", cmd.getOptionValue("c"));
			configuration.setProperty("classifier.multiInstanceClassifier.listOptions", cmd.getOptionValue("o"));

		} else if (cmd.getOptionValue("l").equals("ml")) {
			configuration.setProperty("classifier[@name]", "miml.classifiers.miml.mimlTOml.MIMLClassifierToML");
			configuration.setProperty("classifier.multiLabelClassifier[@name]", cmd.getOptionValue("c"));
			try {
				mlParams = new ConfigLoader(cmd.getOptionValue("o")).getConfiguration();
			} catch (ConfigurationException e) {
				throw new RuntimeException(e);
			}
			if (cmd.getOptionValue("t").equals("arithmetic"))
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.transformation.mimlTOml.ArithmeticTransformation");
			else if (cmd.getOptionValue("t").equals("geometric"))
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.transformation.mimlTOml.GeometricTransformation");
			else if (cmd.getOptionValue("t").equals("minmax"))
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.transformation.mimlTOml.MinMaxTransformation");
			else if (cmd.getOptionValue("t").equals("medoid")) {
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.transformation.mimlTOml.MedoidTransformation");
				configuration.setProperty("classifier.transformationMethod.k", cmd.getOptionValue("k"));
			}
			else if (cmd.getOptionValue("t").equals("means")) {
				configuration.setProperty("classifier.transformationMethod[@name]", "miml.transformation.mimlTOml.KMeansTransformation");
				configuration.setProperty("classifier.transformationMethod.k", cmd.getOptionValue("k"));
			}
		} else if (cmd.getOptionValue("l").equals("miml")) {
			configuration.setProperty("classifier[@name]", cmd.getOptionValue("c"));
			try {
				mlParams = new ConfigLoader(cmd.getOptionValue("o")).getConfiguration();
				for (final Iterator<String> keys = mlParams.getKeys(); keys.hasNext();) {
					final String key = keys.next();
					configuration.addProperty("classifier." + key, mlParams.getProperty(key));
				}
			} catch (ConfigurationException e) {
				throw new RuntimeException(e);
			}
		}

		configuration.setProperty("evaluator.data.trainFile", cmd.getOptionValue("a"));
		configuration.setProperty("evaluator.data.testFile", cmd.getOptionValue("e"));
		configuration.setProperty("evaluator.data.xmlFile", cmd.getOptionValue("x"));
		configuration.setProperty("evaluator.clusWorkingDir", "clusdata");
		Path path = Paths.get(cmd.getOptionValue("a"));
		String clusDataset = path.getFileName().toString();
		configuration.setProperty("evaluator.clusDataset", clusDataset);

		configuration.setProperty("report.fileName", cmd.getOptionValue("r"));

		try {
			IMIMLClassifier mimlExp;
			if (cmd.getOptionValue("l").equals("ml")) {
				assert mlParams != null;
				Params params = Utils.readMultiLabelLearnerParams(mlParams);
				mimlExp = loader.loadClassifier(params);
			}
			else {
				mimlExp = loader.loadClassifier();
			}
			IEvaluator<?> mimlEvaluator = loader.loadEvaluator();
			mimlEvaluator.runExperiment(mimlExp);
			ExperimentReport mimlReport = (ExperimentReport) loader.loadReport();
			mimlReport.toJson(mimlEvaluator);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
