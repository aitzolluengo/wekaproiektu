package Iragarpenak;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Iragarpenak {
    public static void main(String[] args) {
        try {
            if (args.length < 3) {
                System.out.println("Uso: java Iragarpenak <modelo> <datos_test> <OutputFile>");
                return;
            }

            String modelPath = args[0];
            String dataPath = args[1];
            String outputFilePath = args[2];

            Classifier model = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("Modelo cargado desde: " + modelPath);

            DataSource source = new DataSource(dataPath);
            Instances testData = source.getDataSet();

            testData.setClassIndex(testData.numAttributes() - 1);

            System.out.println("Clasificando " + testData.numInstances() + " instancias...");

            FileWriter fileWriter = new FileWriter(outputFilePath);
            PrintWriter printWriter = new PrintWriter(fileWriter);

            for (int i = 0; i < testData.numInstances(); i++) {
                Instance instance = testData.instance(i);
                double prediction = model.classifyInstance(instance);

                String classLabel = (prediction == 0) ? "pos" : "neg";

                printWriter.println("Instancia " + (i + 1) + ": " + classLabel + " (" + (int) prediction + ")");
            }


            printWriter.close();
            System.out.println("Predicciones guardadas en: " + outputFilePath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
