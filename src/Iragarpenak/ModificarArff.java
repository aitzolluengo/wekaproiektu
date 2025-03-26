package Iragarpenak;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.*;
import java.util.ArrayList;

public class ModificarArff {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Uso: java ModificarArff <input.arff> <output.arff>");
            return;
        }

        String inputFile = args[0];
        String outputFile = args[1];

        // Cargar el dataset desde el archivo ARFF
        DataSource source = new DataSource(inputFile);
        Instances data = source.getDataSet();

        // Definir el nuevo atributo de clase (positivo o negativo)
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("pos");
        classValues.add("neg");
        Attribute classAttribute = new Attribute("'@@class@@'", classValues);

        // Agregar la clase al dataset
        data.insertAttributeAt(classAttribute, data.numAttributes());

        // Asignar valores "?" (missing) a todas las instancias en la columna de clase
        for (int i = 0; i < data.numInstances(); i++) {
            data.instance(i).setMissing(data.numAttributes() - 1);
        }

        // Establecer el atributo de clase
        data.setClassIndex(data.numAttributes() - 1);

        // Guardar el nuevo dataset en un archivo ARFF
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        writer.write(data.toString());
        writer.close();

        System.out.println("Archivo ARFF modificado guardado en: " + outputFile);
    }
}
