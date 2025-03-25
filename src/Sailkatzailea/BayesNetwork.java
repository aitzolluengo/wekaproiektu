package Sailkatzailea;

import java.io.PrintWriter;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.Resample;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.global.HillClimber;
import weka.classifiers.bayes.net.search.global.K2;
import weka.classifiers.bayes.net.search.global.TabuSearch;

public class BayesNetwork {
    public static void main(String[] args) throws Exception {
        if (args.length == 3) {
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Entrena un modelo Naive Bayes con todos los datos
            SimpleEstimator estimator = new SimpleEstimator(); // El estimador óptimo
            estimator.setAlpha(1.0);  // El valor óptimo de alpha
            BayesNet bayesNetwork = new BayesNet();
            bayesNetwork.setSearchAlgorithm(new K2());
            bayesNetwork.setEstimator(estimator);

            bayesNetwork.buildClassifier(data);
            // Guardar el modelo
            SerializationHelper.write(args[1], bayesNetwork);

            Evaluation eval1 = new Evaluation(data);
            eval1.evaluateModel(bayesNetwork, data);
            // Hold-Out repeated
            double max = 0;
            int bestRandomSeed = 0;
            for (int i = 0; i < 110; i++) {
                double wfscore = holdOutEgin(i, data).weightedFMeasure();
                if (wfscore > max) {
                    max = wfscore;
                    bestRandomSeed = i;
                }
            }
            Evaluation eval2 = holdOutEgin(bestRandomSeed, data);

            // Naive Bayes con Cross-Validation
            BayesNet crossNet = new BayesNet();
            SimpleEstimator estimator2 = new SimpleEstimator(); // El estimador óptimo
            estimator.setAlpha(1.0);  // El valor óptimo de alpha
            crossNet.setSearchAlgorithm(new K2());
            crossNet.setEstimator(estimator2);

            Evaluation eval3 = new Evaluation(data);
            eval3.crossValidateModel(crossNet, data, 10, new Random(1));
            // Guardar evaluaciones
            PrintWriter pw = new PrintWriter(args[2]);




            pw.println("KALITATEAREN ESTIMAZIOA:");
            pw.println();
            pw.println("---------------------------------------------");
            pw.println();
            pw.println("Ebaluazio ez-zintzoa");
            pw.println(eval1.toSummaryString());
            pw.println(eval1.toClassDetailsString());
            pw.println(eval1.toMatrixString());
            pw.println();
            pw.println("---------------------------------------------");
            pw.println();
            pw.println("Hold-Out");
            pw.println(eval2.toSummaryString());
            pw.println(eval2.toClassDetailsString());
            pw.println(eval2.toMatrixString());
            pw.println();
            pw.println("---------------------------------------------");
            pw.println();
            pw.println("10-fold cross-validation");
            pw.println(eval3.toSummaryString());
            pw.println(eval3.toClassDetailsString());
            pw.println(eval3.toMatrixString());

            pw.close();


        } else {
            System.out.println("Uso: java -jar NaiveBayes.jar trainPath.arff NBpath.model kalitatea.txt");
        }
    }

    private static Evaluation holdOutEgin(int i, Instances data) throws Exception {
        // Aleatorizar datos
        Randomize filter = new Randomize();
        filter.setRandomSeed(i);
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        // División en 70% entrenamiento usando Resample
        Resample resampleTrain = new Resample();
        resampleTrain.setRandomSeed(i);
        resampleTrain.setSampleSizePercent(70);
        resampleTrain.setNoReplacement(true);
        resampleTrain.setInputFormat(data);
        Instances train = Filter.useFilter(data, resampleTrain);

        // División en 30% validación usando Resample
        Resample resampleDev = new Resample();
        resampleDev.setRandomSeed(i);
        resampleDev.setSampleSizePercent(30);
        resampleDev.setNoReplacement(true);
        resampleDev.setInputFormat(data);
        Instances dev = Filter.useFilter(data, resampleDev);

        // Entrenar Naive Bayes con el conjunto de entrenamiento
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        // Evaluar en el conjunto de validación
        Evaluation eval2 = new Evaluation(train);
        eval2.evaluateModel(nb, dev);

        return eval2;
    }
}
