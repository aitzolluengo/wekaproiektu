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
            // Verifica que se pasaron los argumentos correctos
            if (args.length < 2) {
                System.out.println("Uso: java Iragarpenak <modelo> <datos_test>");
                return;
            }

            // Obtener las rutas del modelo y los datos desde los argumentos
            String modelPath = args[0];
            String dataPath = args[1];

            // Cargar el modelo entrenado
            Classifier model = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("Modelo cargado desde: " + modelPath);

            // Cargar el conjunto de datos de prueba (sin clase)
            DataSource source = new DataSource(dataPath);
            Instances testData = source.getDataSet();

            // Asegurar que el dataset tiene una clase establecida
            testData.setClassIndex(testData.numAttributes() - 1);

            System.out.println("Clasificando " + testData.numInstances() + " instancias...");

            // Abrir archivo para guardar las predicciones
            String outputFilePath = "predicciones.txt";
            FileWriter fileWriter = new FileWriter(outputFilePath);
            PrintWriter printWriter = new PrintWriter(fileWriter);

            // Clasificar cada instancia y guardar resultados
            for (int i = 0; i < testData.numInstances(); i++) {
                Instance instance = testData.instance(i);
                double prediction = model.classifyInstance(instance);
                printWriter.println(prediction); // Guardar solo la predicción
            }

            // Cerrar el archivo
            printWriter.close();
            System.out.println("Predicciones guardadas en: " + outputFilePath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
