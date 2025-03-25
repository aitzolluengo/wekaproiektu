package Sailkatzailea;
import weka.classifiers.Evaluation;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.global.HillClimber;
import weka.classifiers.bayes.net.search.global.K2;
import weka.classifiers.bayes.net.search.global.TabuSearch;
import weka.classifiers.functions.LinearRegression; // Cambio aquí
import weka.classifiers.functions.MultilayerPerceptron; // Cambio aquí
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;
import weka.classifiers.bayes.BayesNet.*;
/**
 * The Class ParametroenBilaketa.
 */
public class ParametroenBilaketa {
    private static PrintWriter pw;


    public static void main(String[] args) throws Exception {
        if (args.length == 3) { // Cambiamos a 3 argumentos
            String algoritmo = args[2]; // El tercer argumento es el algoritmo a usar
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            FileWriter filewriter = new FileWriter(args[1]);
            pw = new PrintWriter(filewriter);

            data.randomize(new java.util.Random(1));

            RemovePercentage RP = new RemovePercentage();
            RP.setPercentage(70);
            RP.setInputFormat(data);

            RP.setInvertSelection(true);
            Instances train = Filter.useFilter(data, RP);

            RP.setInputFormat(data);
            RP.setInvertSelection(false);
            Instances test = Filter.useFilter(data, RP);

            int i = klaseminoritarioa(data);

            if (algoritmo.equalsIgnoreCase("Bayes Network")) {
                // Bayes Network
                SearchAlgorithm[] searchAlgorithms = {new K2(), new HillClimber(), new TabuSearch()};
                double[] alphaValues = {0.1, 0.5, 1.0}; // Valores de suavizado para SimpleEstimator

                double maxAccuracy = 0.0;
                SearchAlgorithm bestSearchAlgorithm = null;
                double bestAlpha = 0.1;

                for (SearchAlgorithm searchAlgorithm : searchAlgorithms) {
                    for (double alpha : alphaValues) {
                        BayesNet bnTemp = new BayesNet();
                        bnTemp.setSearchAlgorithm(searchAlgorithm);

                        // Crear el estimador y establecer su alpha
                        SimpleEstimator estimator = new SimpleEstimator();
                        estimator.setAlpha(alpha);  // Aquí se usa setAlpha() en lugar de pasar un double directamente
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

                System.out.println("Best SearchAlgorithm: " + bestSearchAlgorithm.getClass().getSimpleName());
                System.out.println("Best Alpha: " + bestAlpha);
                System.out.println("Max Accuracy: " + maxAccuracy);

            } else if (algoritmo.equalsIgnoreCase("Naive Bayes")) {
                // Optimización de parámetros para Naive Bayes

                pw.println();
                pw.println("MLP parametro ekorketa");
                pw.println("3 parametro optimizatuko ditugu:");
                pw.println("1- hiddenLayers");
                pw.println("2- learningRate");
                pw.println("3- trainingTime");
                pw.println("Ebaluazio metrika: Klase minoritarioaren fMeasure");
                pw.println();


                boolean[] kernelEstimatorOptions = {true, false};
                boolean[] discretizationOptions = {true, false};

                double maxAccuracy = 0.0;
                boolean bestKernelEstimator = false;
                boolean bestDiscretization = false;

                for (boolean useKernelEstimator : kernelEstimatorOptions) {
                    for (boolean useDiscretization : discretizationOptions) {
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

                pw.println("Best KernelEstimator: " + bestKernelEstimator);
                pw.println("Best Discretization: " + bestDiscretization);
                pw.println("Max Accuracy: " + maxAccuracy);


            } else {
                System.out.println("Algoritmo no válido. Usa 'LinearRegression' o 'MLP'.");
                return;
            }

            pw.close();
        } else {
            System.out.println(args.length + " parametro sartu dituzu");
            System.out.println("3 parametro sartu behar dituzu!");
            System.out.println("java -jar ParametroenBilaketa.jar data.arff parametroak.txt LinearRegression|MLP");
        }
    }

    /**
     * Klaseminoritarioa.
     *
     * @return the int
     * @throws Exception the exception
     */
    private static int klaseminoritarioa(Instances data) throws Exception {
        int minfreq=0;
        int minclassindex=0;
        for (int i=0; i<data.classAttribute().numValues();i++){
            String value= data.classAttribute().value(i);
            int freq = data.attributeStats(data.classIndex()).nominalCounts[i];
            int min = 0; //Hemen klase minoritarioaren posizioa gordeko da
            if (freq < minfreq) {
                minfreq = freq;
                minclassindex = i;
            }
            }
            pw.println(minclassindex + "" + minfreq);
            return minclassindex;
    }
}


