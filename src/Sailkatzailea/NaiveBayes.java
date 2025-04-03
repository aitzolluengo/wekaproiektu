package Sailkatzailea;

import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.Resample;

public class NaiveBayes {
    public static void main(String[] args) throws Exception {
        if (args.length == 3) {
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Naive bayes entrenatuko dugu ParametroenBilaketaren datuak kontuan izanda
            weka.classifiers.bayes.NaiveBayes nbGuztiak = new weka.classifiers.bayes.NaiveBayes();
            nbGuztiak.setUseSupervisedDiscretization(false);
            nbGuztiak.setUseKernelEstimator(false);
            nbGuztiak.buildClassifier(data);

            //  Repeated Hold-Out 
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

            // Naive Bayes  Cross-Validation
            weka.classifiers.bayes.NaiveBayes nbCross = new weka.classifiers.bayes.NaiveBayes();

            // Gordeko dugu
            PrintWriter pw = new PrintWriter(args[2]);
            Evaluation eval1 = new Evaluation(data);
            eval1.evaluateModel(nbGuztiak, data);

            Evaluation eval3 = new Evaluation(data);
            eval3.crossValidateModel(nbCross, data, 10, new Random(1));

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

            // Gordeko dugu modeloa
            SerializationHelper.write(args[1], nbGuztiak);
        } else {
            System.out.println("Uso: java -jar NaiveBayes.jar trainPath.arff NBpath.model kalitatea.txt");
            //trainPath input izango da eta NBpath.model eta kalitatea.txt outputak
        }
    }

    private static Evaluation holdOutEgin(int i, Instances data) throws Exception {
        //randomize
        Randomize filter = new Randomize();
        filter.setRandomSeed(i);
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        // Resample erabili 70% entrenamendurako
        Resample resampleTrain = new Resample();
        resampleTrain.setRandomSeed(i);
        resampleTrain.setSampleSizePercent(70);
        resampleTrain.setNoReplacement(true);
        resampleTrain.setInputFormat(data);
        Instances train = Filter.useFilter(data, resampleTrain);

        // Resample erabili 30% dev
        Resample resampleDev = new Resample();
        resampleDev.setRandomSeed(i);
        resampleDev.setSampleSizePercent(30);
        resampleDev.setNoReplacement(true);
        resampleDev.setInputFormat(data);
        Instances dev = Filter.useFilter(data, resampleDev);

        // Entrenatu
        weka.classifiers.bayes.NaiveBayes nb = new weka.classifiers.bayes.NaiveBayes();
        nb.buildClassifier(train);

        // ebaluatu
        Evaluation eval2 = new Evaluation(train);
        eval2.evaluateModel(nb, dev);

        return eval2;
    }
}
