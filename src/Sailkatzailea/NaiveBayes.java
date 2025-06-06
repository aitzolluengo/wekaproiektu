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
        if (args.length == 4) {  // Ahora acepta 4 parámetros (el número de repeticiones)
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Naive bayes entrenatuko dugu ParametroenBilaketaren datuak kontuan izanda
            weka.classifiers.bayes.NaiveBayes nbGuztiak = new weka.classifiers.bayes.NaiveBayes();
            nbGuztiak.setUseSupervisedDiscretization(false);
            nbGuztiak.setUseKernelEstimator(false);
            nbGuztiak.buildClassifier(data);

            // Iterazio kopurua lortu
            int repeticiones = Integer.parseInt(args[3]);

            // Repeated Hold-Out
            double averageFMeasure = 0;
            double averagePrecision = 0;
            double averageRecall = 0;
            double averageAccuracy = 0;

            for (int j = 0; j < repeticiones; j++) {
                Evaluation eval = holdOutEgin(j, data);
                averageFMeasure += eval.weightedFMeasure();
                averagePrecision += eval.weightedPrecision();
                averageRecall += eval.weightedRecall();
                averageAccuracy += eval.pctCorrect();
            }

            averageFMeasure /= repeticiones;
            averagePrecision /= repeticiones;
            averageRecall /= repeticiones;
            averageAccuracy /= repeticiones;

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
            pw.println("Hold-Out Promedios (Repeticiones: " + repeticiones + ")");
            pw.println("F-Measure promedio: " + averageFMeasure);
            pw.println("Precision promedio: " + averagePrecision);
            pw.println("Recall promedio: " + averageRecall);
            pw.println("Accuracy promedio: " + averageAccuracy);
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
            System.out.println("Uso: java -jar NaiveBayes.jar trainPath.arff NBpath.model kalitatea.txt repeticiones");
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
        nb.setUseSupervisedDiscretization(false);
        nb.setUseKernelEstimator(false);
        nb.buildClassifier(train);

        // ebaluatu
        Evaluation eval2 = new Evaluation(train);
        eval2.evaluateModel(nb, dev);

        return eval2;
    }
}
