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

package miml.run;

import miml.classifiers.miml.IMIMLClassifier;
import miml.core.ConfigLoader;
import miml.evaluation.IEvaluator;
import miml.report.IReport;
import org.apache.commons.cli.*;
import org.apache.commons.configuration2.Configuration;

import java.io.File;
import java.io.IOException;
import java.util.Date;

public class RunExperiment {

	/**
	 * The main method to configure and run an algorithm.
	 *
	 * @param args the arguments(route of config file with the option -c)
	 */
	public static void main(String[] args) {
		String[] configFiles = new String[] {
				"tomi-br-citationknn-text.config",
				"tomi-br-simplemi-a-text.config",
				"tomi-br-simplemi-g-text.config",
				"tomi-br-simplemi-mm-text.config",
				"tomi-br-miwrapper-a-text.config",
				"tomi-br-miwrapper-g-text.config",
				"tomi-br-miwrapper-mp-text.config",
				"tomi-br-miboost-text.config",
				"tomi-br-milr-text.config",
				"tomi-br-miri-text.config",
				"tomi-br-miti-text.config",
				"tomi-br-mismo-text.config",
				"tomi-br-misvm-text.config",
				"tomi-lr-citationknn-text.config",
				"tomi-lr-simplemi-a-text.config",
				"tomi-lr-simplemi-g-text.config",
				"tomi-lr-simplemi-mm-text.config",
				"tomi-lr-miwrapper-a-text.config",
				"tomi-lr-miwrapper-g-text.config",
				"tomi-lr-miwrapper-mp-text.config",
				"toml-br-a-text.config",
				"toml-br-g-text.config",
				"toml-br-mm-text.config",
				"toml-lp-a-text.config",
				"toml-lp-g-text.config",
				"toml-lp-mm-text.config",
				"toml-rpc-a-text.config",
				"toml-rpc-g-text.config",
				"toml-rpc-mm-text.config",
				"toml-clr-a-text.config",
				"toml-clr-g-text.config",
				"toml-clr-mm-text.config",
				"toml-ps-a-text.config",
				"toml-ps-g-text.config",
				"toml-ps-mm-text.config",
				"toml-cc-a-text.config",
				"toml-cc-g-text.config",
				"toml-cc-mm-text.config",
				"toml-mlstacking-a-text.config",
				"toml-mlstacking-g-text.config",
				"toml-mlstacking-mm-text.config",
				"toml-homer-a-text.config",
				"toml-homer-g-text.config",
				"toml-homer-mm-text.config",
				"toml-rakel-a-text.config",
				"toml-rakel-g-text.config",
				"toml-rakel-mm-text.config",
				"toml-eps-a-text.config",
				"toml-eps-g-text.config",
				"toml-eps-mm-text.config",
				"toml-ecc-a-text.config",
				"toml-ecc-g-text.config",
				"toml-ecc-mm-text.config",
				"tomi-br-mdd-text.config",
				"tomi-br-midd-text.config",
				"tomi-br-mioptimalball-text.config",
		};
		String configPath = "configExp/";
		String datasetsPath = null;
		try {
			datasetsPath = new File("").getCanonicalFile().getParent() + "/datasets/";
		} catch (IOException e) {
			e.printStackTrace();
		}

		Options options = new Options();

		Option confStartArg = new Option("c", true, "configuration to start from");
		options.addOption(confStartArg);

		Option confEndArg = new Option("e", true, "configuration to end");
		options.addOption(confEndArg);

		Option foldStartArg = new Option("f", true, "fold to start from");
		options.addOption(foldStartArg);

		Option datasetsArg = new Option("d", "datasets to compute");
		datasetsArg.setRequired(true);
		datasetsArg.setArgs(Option.UNLIMITED_VALUES);
		options.addOption(datasetsArg);

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

		int confStart = 0;
		int offset = Integer.parseInt(cmd.getOptionValue("c", "0"));
		int confEnd = Integer.parseInt(cmd.getOptionValue("e", String.valueOf(configFiles.length)));
		int foldStart = Integer.parseInt(cmd.getOptionValue("f", "1"));
		String[] datasets = cmd.getOptionValues("d");

		for (String dataset : datasets) {
			for (int i = confStart + offset; i < confEnd; ++i) {
				int jIni = 1;
				if (i == confStart) jIni = foldStart;
				for (int j = jIni; j <= 5; ++j) {
					System.out.println(new Date() + ": Starting experiment " + configFiles[i] + ", dataset " + dataset + ", partition " + j);
					try {
						ConfigLoader loader = new ConfigLoader(configPath + configFiles[i]);

						Configuration configuration = loader.getConfiguration();
						configuration.setProperty("evaluator.data.trainFile", datasetsPath + dataset + "/5-folds/rounds/miml_" + dataset + "_iterative_5_train_" + j + ".arff");
						configuration.setProperty("evaluator.data.testFile", datasetsPath + dataset + "/5-folds/rounds/miml_" + dataset + "_iterative_5_test_" + j + ".arff");
						configuration.setProperty("evaluator.data.xmlFile", datasetsPath + dataset + "/5-folds/miml_" + dataset + ".xml");
						configuration.setProperty("report.fileName", "results2/" + dataset + "-res.csv");

						IMIMLClassifier classifier = loader.loadClassifier();
						IEvaluator<?> evaluator = loader.loadEvaluator();
						evaluator.runExperiment(classifier);
						IReport report = loader.loadReport();
						report.toCSV(evaluator);
					} catch (Exception e) {
						e.printStackTrace();
					}
					System.out.println(new Date() + ": Ending Experiment");
				}
			}
		}
	}
}
