package Classifier;
import weka.classifiers.Evaluation;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import weka.classifiers.functions.LinearRegression; // Cambio aquí
import weka.classifiers.functions.MultilayerPerceptron; // Cambio aquí
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;
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
            data = convertirClaseNominalANumerica(data);
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

            if (algoritmo.equalsIgnoreCase("LinearRegression")) {
                // Linear Regression
                LinearRegression lr = new LinearRegression();
                lr.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(lr,test);
                System.out.println("aaaaa");
                pw.println("Linear Regression - Ebaluazio Metrikak:");
                pw.println(eval.toSummaryString());
                //pw.println(eval.toClassDetailsString());
                //pw.println(eval.toMatrixString());
            } else if (algoritmo.equalsIgnoreCase("MLP")) {
                // Optimización de parámetros para MLP
                double maximoa = 0.0;
                String bestHiddenLayers = "";
                double bestLearningRate = 0.0;
                pw.println();
                pw.println("MLP parametro ekorketa");
                pw.println("3 parametro optimizatuko ditugu:");
                pw.println("1- hiddenLayers");
                pw.println("2- learningRate");
                pw.println("3- trainingTime");
                pw.println("Ebaluazio metrika: Klase minoritarioaren fMeasure");
                pw.println();

                int i = klaseminoritarioa(data);

                // Rango de valores para los parámetros
                String[] hiddenLayersOptions = {"a", "t", "i", "o", "0"}; // Ejemplo de opciones
                double[] learningRates = {0.1, 0.3, 0.5, 0.7}; // Ejemplo de tasas de aprendizaje

                for (String hiddenLayers : hiddenLayersOptions) {
                    for (double learningRate : learningRates) {
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setHiddenLayers(hiddenLayers);
                        mlp.setLearningRate(learningRate);
                        mlp.setTrainingTime(10);
                        System.out.println(hiddenLayers);
                        System.out.println(learningRate);
                        mlp.buildClassifier(train);
                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(mlp,test);

                        if (eval.fMeasure(i) > maximoa) {
                            maximoa = eval.fMeasure(i);
                            System.out.println(maximoa);
                            bestHiddenLayers = hiddenLayers;
                            bestLearningRate = learningRate;
                        }

                    }
                }

                pw.println("Hidden Layers hoberena:");
                pw.println(bestHiddenLayers);
                pw.println("Learning Rate hoberena:");
                pw.println(bestLearningRate);
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
    private static int klaseminoritarioa(Instances data) throws Exception {
        int minfreq=Integer.MAX_VALUE;
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

    public static Instances convertirClaseNominalANumerica(Instances data) throws Exception {
        // Crear nueva estructura sin la clase nominal original
        ArrayList<Attribute> atributos = new ArrayList<>();

        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex()) {
                atributos.add(data.attribute(i));
            }
        }

        // Añadir atributo de clase numérica
        atributos.add(new Attribute("class")); // numérico por defecto

        // Crear nuevo dataset
        Instances nuevoData = new Instances("data_numerico", atributos, data.numInstances());
        nuevoData.setClassIndex(nuevoData.numAttributes() - 1);

        // Copiar datos y transformar clase
        for (int i = 0; i < data.numInstances(); i++) {
            double[] vals = new double[nuevoData.numAttributes()];

            int k = 0;
            for (int j = 0; j < data.numAttributes(); j++) {
                if (j == data.classIndex()) continue;
                vals[k++] = data.instance(i).value(j);
            }

            // Transformar clase nominal a numérica: spam = 1, ham = 0
            String claseOriginal = data.instance(i).stringValue(data.classIndex());
            vals[k] = claseOriginal.equalsIgnoreCase("spam") ? 1.0 : 0.0;

            nuevoData.add(new DenseInstance(1.0, vals));
        }

        return nuevoData;
    }
}
