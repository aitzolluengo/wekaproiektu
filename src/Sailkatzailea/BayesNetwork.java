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
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.global.K2;


public class BayesNetwork {
    public static void main(String[] args) throws Exception {
        if (args.length == 4) {
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            int repeticiones = Integer.parseInt(args[3]);

            // Naive Bayes eredu bat trebatzen du datu guztiekin
            SimpleEstimator estimator = new SimpleEstimator(); // Estimatzaile optimoa
            estimator.setAlpha(1.0);  // Alpha balio optimoa
            BayesNet bayesNetwork = new BayesNet();
            bayesNetwork.setSearchAlgorithm(new K2());
            bayesNetwork.setEstimator(estimator);

            bayesNetwork.buildClassifier(data);
            // Eredua gordetzen du
            SerializationHelper.write(args[1], bayesNetwork);

            Evaluation eval1 = new Evaluation(data);
            eval1.evaluateModel(bayesNetwork, data);

            // Bayes Network Cross-Validation erabiliz
            bayesNetwork.buildClassifier(data);
            Evaluation eval2 = new Evaluation(data);
            eval2.crossValidateModel(bayesNetwork, data, 10, new Random(1));

            // Ebaluazioak gordetzen ditu
            PrintWriter pw = new PrintWriter(args[2]);

            // Repeated Hold-Out
            double averageFMeasure = 0;
            double averagePrecision = 0;
            double averageRecall = 0;
            double averageAccuracy = 0;

            for (int j = 0; j < repeticiones; j++) {
                Evaluation eval = holdOutEgin(data);
                averageFMeasure += eval.weightedFMeasure();
                averagePrecision += eval.weightedPrecision();
                averageRecall += eval.weightedRecall();
                averageAccuracy += eval.pctCorrect();
            }

            averageFMeasure /= repeticiones;
            averagePrecision /= repeticiones;
            averageRecall /= repeticiones;
            averageAccuracy /= repeticiones;

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
            pw.println("Repeated Hold-Out");
            pw.println("Batez besteko F-Measure: " + averageFMeasure);
            pw.println("Batez besteko Precision: " + averagePrecision);
            pw.println("Batez besteko Recall: " + averageRecall);
            pw.println("Batez besteko Accuracy: " + averageAccuracy);
            pw.println();
            pw.println("---------------------------------------------");
            pw.println();
            pw.println("10-fold cross-validation");
            pw.println(eval2.toSummaryString());
            pw.println(eval2.toClassDetailsString());
            pw.println(eval2.toMatrixString());

            pw.close();
        } else {
            System.out.println("Erabilera: java -jar NaiveBayes.jar trainPath.arff NBpath.model kalitatea.txt repeticiones");
        }
    }

    private static Evaluation holdOutEgin(Instances data) throws Exception {
        // Datuak ausaz ordenatzen ditu
        Randomize filter = new Randomize();
        filter.setRandomSeed(new Random().nextInt());
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        // Datuak bi zatitan banatzen ditu: %70 entrenamendurako
        Resample resampleTrain = new Resample();
        resampleTrain.setRandomSeed(new Random().nextInt());
        resampleTrain.setSampleSizePercent(70);
        resampleTrain.setNoReplacement(true);
        resampleTrain.setInputFormat(data);
        Instances train = Filter.useFilter(data, resampleTrain);

        // %30 balidaziorako
        Resample resampleDev = new Resample();
        resampleDev.setRandomSeed(new Random().nextInt());
        resampleDev.setSampleSizePercent(30);
        resampleDev.setNoReplacement(true);
        resampleDev.setInputFormat(data);
        Instances dev = Filter.useFilter(data, resampleDev);

        // Naive Bayes eredu bat trebatzen du datu guztiekin
        SimpleEstimator estimator = new SimpleEstimator(); // Estimatzaile optimoa
        estimator.setAlpha(1.0);  // Alpha balio optimoa
        BayesNet bayesNetworkk = new BayesNet();
        bayesNetworkk.setSearchAlgorithm(new K2());
        bayesNetworkk.setEstimator(estimator);

        bayesNetworkk.buildClassifier(train);

        // Eredua balidazio multzoan ebaluatzen du
        Evaluation eval2 = new Evaluation(train);
        eval2.evaluateModel(bayesNetworkk, dev);

        return eval2;
    }
}
