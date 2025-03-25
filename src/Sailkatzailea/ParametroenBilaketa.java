package Sailkatzailea;

import weka.classifiers.Evaluation;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.global.HillClimber;
import weka.classifiers.bayes.net.search.global.K2;
import weka.classifiers.bayes.net.search.global.TabuSearch;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;

public class ParametroenBilaketa {
    private static PrintWriter pw;

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println(args.length + " parametro sartu dituzu");
            System.out.println("3 parametro sartu behar dituzu!");
            System.out.println("java -jar ParametroenBilaketa.jar data.arff parametroak.txt Bayes Network|Naive Bayes");
            return;
        }

        String algoritmo = args[2];
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        pw = new PrintWriter(new FileWriter(args[1]));

        data.randomize(new java.util.Random(1));

        RemovePercentage RP = new RemovePercentage();
        RP.setPercentage(70);
        RP.setInvertSelection(true);
        RP.setInputFormat(data);
        Instances train = Filter.useFilter(data, RP);

        RP.setInvertSelection(false);
        Instances test = Filter.useFilter(data, RP);

        int i = klaseminoritarioa(data);

        if (algoritmo.equalsIgnoreCase("Bayes Network")) {
            optimizeBayesNetwork(train, test);
        } else if (algoritmo.equalsIgnoreCase("Naive Bayes")) {
            optimizeNaiveBayes(train, test, i);
        } else {
            System.out.println("Algoritmo no válido. Usa 'Bayes Network' o 'Naive Bayes'.");
        }

        pw.flush();
        pw.close();
    }

    private static void optimizeBayesNetwork(Instances train, Instances test) throws Exception {
        SearchAlgorithm[] searchAlgorithms = {new K2(), new HillClimber(), new TabuSearch()};
        double[] alphaValues = {0.1, 0.5, 1.0};

        double maxAccuracy = 0.0;
        SearchAlgorithm bestSearchAlgorithm = null;
        double bestAlpha = 0.1;

        for (SearchAlgorithm searchAlgorithm : searchAlgorithms) {
            for (double alpha : alphaValues) {
                BayesNet bnTemp = new BayesNet();
                bnTemp.setSearchAlgorithm(searchAlgorithm);

                SimpleEstimator estimator = new SimpleEstimator();
                estimator.setAlpha(alpha);
                bnTemp.setEstimator(estimator);

                bnTemp.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(bnTemp, test);

                if (eval.fMeasure(1) > maxAccuracy) {
                    maxAccuracy = eval.fMeasure(1);
                    bestSearchAlgorithm = searchAlgorithm;
                    bestAlpha = alpha;
                }
            }
        }

        String result = "Best SearchAlgorithm: " + bestSearchAlgorithm.getClass().getSimpleName() +
                "\nBest Alpha: " + bestAlpha +
                "\nMax Accuracy: " + maxAccuracy;
        System.out.println(result);
        pw.println(result);
    }

    private static void optimizeNaiveBayes(Instances train, Instances test, int i) throws Exception {
        boolean[] kernelEstimatorOptions = {true, false};
        boolean[] discretizationOptions = {true, false};

        double maxAccuracy = 0.0;
        boolean bestKernelEstimator = false;
        boolean bestDiscretization = false;

        for (boolean useKernelEstimator : kernelEstimatorOptions) {
            for (boolean useDiscretization : discretizationOptions) {
                // Evita el error: No se pueden usar ambos al mismo tiempo
                if (useKernelEstimator && useDiscretization) {
                    continue;  // Saltamos esta combinación
                }

                NaiveBayes nb = new NaiveBayes();
                nb.setUseKernelEstimator(useKernelEstimator);
                nb.setUseSupervisedDiscretization(useDiscretization);

                nb.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(nb, test);

                if (eval.fMeasure(i) > maxAccuracy) {
                    maxAccuracy = eval.fMeasure(i);
                    bestKernelEstimator = useKernelEstimator;
                    bestDiscretization = useDiscretization;
                }
            }
        }

        String result = "Best KernelEstimator: " + bestKernelEstimator +
                "\nBest Discretization: " + bestDiscretization +
                "\nMax Accuracy: " + maxAccuracy;
        System.out.println(result);
        pw.println(result);
    }

    private static int klaseminoritarioa(Instances data) {
        int minfreq = Integer.MAX_VALUE;
        int minclassindex = 0;

        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            int freq = data.attributeStats(data.classIndex()).nominalCounts[i];
            if (freq < minfreq) {
                minfreq = freq;
                minclassindex = i;
            }
        }

        String result = "Clase minoritaria: " + minclassindex + " Frecuencia: " + minfreq;
        System.out.println(result);
        pw.println(result);

        return minclassindex;
    }
}
