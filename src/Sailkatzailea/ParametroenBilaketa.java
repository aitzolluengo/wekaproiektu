package Sailkatzailea;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression; // Cambio aquí
import weka.classifiers.functions.MultilayerPerceptron; // Cambio aquí
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * The Class ParametroenBilaketa.
 */
public class ParametroenBilaketa {

    private static PrintWriter pw;
    private static Instances data;

    public static void main(String[] args) throws Exception {
        if (args.length == 3) { // Cambiamos a 3 argumentos
            String algoritmo = args[2]; // El tercer argumento es el algoritmo a usar
            DataSource source = new DataSource(args[0]);
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            FileWriter filewriter = new FileWriter(args[1]);
            pw = new PrintWriter(filewriter);

            if (algoritmo.equalsIgnoreCase("LinearRegression")) {
                // Linear Regression
                LinearRegression lr = new LinearRegression();
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(lr, data, 10, new Random(1));

                pw.println("Linear Regression - Ebaluazio Metrikak:");
                pw.println(eval.toSummaryString());
                pw.println(eval.toClassDetailsString());
                pw.println(eval.toMatrixString());
            } else if (algoritmo.equalsIgnoreCase("MLP")) {
                // Optimización de parámetros para MLP
                double maximoa = 0.0;
                String bestHiddenLayers = "";
                double bestLearningRate = 0.0;
                int bestTrainingTime = 0;

                pw.println();
                pw.println("MLP parametro ekorketa");
                pw.println("3 parametro optimizatuko ditugu:");
                pw.println("1- hiddenLayers");
                pw.println("2- learningRate");
                pw.println("3- trainingTime");
                pw.println("Ebaluazio metrika: Klase minoritarioaren fMeasure");
                pw.println();

                int i = klaseminoritarioa();

                // Rango de valores para los parámetros
                String[] hiddenLayersOptions = {"1", "3", "5"}; // Ejemplo de opciones
                double[] learningRates = {0.1, 0.3, 0.5}; // Ejemplo de tasas de aprendizaje
                int[] trainingTimes = {100, 200, 300}; // Ejemplo de épocas

                for (String hiddenLayers : hiddenLayersOptions) {
                    for (double learningRate : learningRates) {
                        for (int trainingTime : trainingTimes) {
                            MultilayerPerceptron mlp = new MultilayerPerceptron();
                            mlp.setHiddenLayers(hiddenLayers);
                            mlp.setLearningRate(learningRate);
                            mlp.setTrainingTime(trainingTime);

                            Evaluation eval = new Evaluation(data);
                            eval.crossValidateModel(mlp, data, 10, new Random(1));

                            if (eval.fMeasure(i) > maximoa) {
                                maximoa = eval.fMeasure(i);
                                bestHiddenLayers = hiddenLayers;
                                bestLearningRate = learningRate;
                                bestTrainingTime = trainingTime;
                            }
                        }
                    }
                }

                pw.println("Hidden Layers hoberena:");
                pw.println(bestHiddenLayers);
                pw.println("Learning Rate hoberena:");
                pw.println(bestLearningRate);
                pw.println("Training Time hoberena:");
                pw.println(bestTrainingTime);
                pw.println("Klase minoritarioaren fMeasure hoberena:");
                pw.println(maximoa);
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
    private static int klaseminoritarioa() throws Exception {
        //Klase minoritarioaren indizea itzultzen du
        pw.println(data.attribute(data.numAttributes() - 1).name() + " atributu nominala da eta hauek dira ezaugarriak:");
        int[] counts = data.attributeStats(data.numAttributes() - 1).nominalCounts;
        int min = 0; //Hemen klase minoritarioaren posizioa gordeko da
        for (int j = 0; j < counts.length; j++) {
            if (counts[min] > counts[j]) {
                min = j;
            }
            pw.println(data.attribute(data.numAttributes() - 1).value(j) + " -> " + counts[j] + " | Maiztasuna -> " + (float) counts[j] / data.attributeStats(data.numAttributes() - 1).totalCount);
        }
        pw.println("Balio minimoa: " + data.attribute(data.numAttributes() - 1).value(min) + " -> " + counts[min] + "\n");
        return min;
    }
}