package Iragarpenak;
import java.io.*;
import java.util.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ARFFAttributeReducer {

    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Uso: java ARFFAttributeReducer <train.arff> <test_blind.arff> <output.arff>");
            return;
        }

        String trainPath = args[0];
        String testPath = args[1];
        String outputPath = args[2];

        try {
            // Cargar los conjuntos de datos
            DataSource trainSource = new DataSource(trainPath);
            Instances trainData = trainSource.getDataSet();

            DataSource testSource = new DataSource(testPath);
            Instances testData = testSource.getDataSet();

            // Verificar que la clase esté presente
            if (testData.classIndex() < 0) {
                testData.setClassIndex(testData.numAttributes() - 1);
            }

            // Obtener nombres de atributos del test
            ArrayList<String> testAttributes = new ArrayList<>();
            for (int i = 0; i < testData.numAttributes(); i++) {
                testAttributes.add(testData.attribute(i).name());
            }

            // Crear lista de índices de atributos a mantener (incluyendo la clase)
            ArrayList<Integer> attributesToKeep = new ArrayList<>();

            // Buscar coincidencias de nombres de atributos
            for (String attrName : testAttributes) {
                boolean found = false;
                for (int i = 0; i < trainData.numAttributes(); i++) {
                    if (trainData.attribute(i).name().equals(attrName)) {
                        attributesToKeep.add(i);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    System.out.println("Advertencia: Atributo '" + attrName + "' no encontrado en train.arff");
                }
            }

            // Verificar que coincida el número de atributos
            if (attributesToKeep.size() != testData.numAttributes()) {
                System.out.println("Error: No se pudieron encontrar todos los atributos. Esperados: " +
                        testData.numAttributes() + ", Encontrados: " + attributesToKeep.size());
                return;
            }

            // Reducir el conjunto de entrenamiento
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(getInverseIndices(trainData.numAttributes(), attributesToKeep));
            removeFilter.setInputFormat(trainData);
            Instances reducedTrainData = Filter.useFilter(trainData, removeFilter);

            // Verificar el orden de los atributos
            for (int i = 0; i < reducedTrainData.numAttributes(); i++) {
                if (!reducedTrainData.attribute(i).name().equals(testData.attribute(i).name())) {
                    System.out.println("Error: Los atributos no coinciden en orden. Se requiere reordenamiento.");
                    reorderAttributes(reducedTrainData, testData, outputPath);
                    return;
                }
            }

            // Guardar el resultado
            saveARFF(reducedTrainData, outputPath);

            System.out.println("Proceso completado. Archivo reducido guardado en: " + outputPath);
            System.out.println("Atributos originales: " + trainData.numAttributes() +
                    ", Atributos reducidos: " + reducedTrainData.numAttributes());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static int[] getInverseIndices(int totalAttributes, ArrayList<Integer> attributesToKeep) {
        // Crear lista de índices a eliminar (inverso de los que se mantienen)
        ArrayList<Integer> toRemove = new ArrayList<>();
        for (int i = 0; i < totalAttributes; i++) {
            if (!attributesToKeep.contains(i)) {
                toRemove.add(i);
            }
        }

        // Convertir a array primitivo
        int[] result = new int[toRemove.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = toRemove.get(i);
        }
        return result;
    }

    private static void reorderAttributes(Instances data, Instances reference, String outputPath) throws Exception {
        // Reordenar atributos para que coincidan con el conjunto de referencia
        ArrayList<Attribute> newAttributes = new ArrayList<>();

        // Agregar atributos en el orden del conjunto de referencia
        for (int i = 0; i < reference.numAttributes(); i++) {
            Attribute refAttr = reference.attribute(i);
            Attribute foundAttr = data.attribute(refAttr.name());
            if (foundAttr == null) {
                throw new Exception("Atributo " + refAttr.name() + " no encontrado en los datos");
            }
            newAttributes.add(foundAttr);
        }

        // Crear nuevo conjunto de datos con atributos reordenados
        Instances reorderedData = new Instances(data.relationName(), newAttributes, data.numInstances());

        // Copiar los datos
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] values = new double[reference.numAttributes()];

            for (int j = 0; j < reference.numAttributes(); j++) {
                Attribute refAttr = reference.attribute(j);
                Attribute dataAttr = data.attribute(refAttr.name());
                values[j] = inst.value(dataAttr);
            }

            reorderedData.add(new DenseInstance(inst.weight(), values));
        }

        // Establecer índice de clase
        reorderedData.setClassIndex(reference.classIndex());

        // Guardar
        saveARFF(reorderedData, outputPath);
    }

    private static void saveARFF(Instances data, String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(data.toString());
        writer.flush();
        writer.close();
    }
}