package Iragarpenak;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import java.io.*;
import java.util.*;

public class ModificarArff {
    public static void main(String[] args) {
        try {
            validateArguments(args);
            int predictorsCount = Integer.parseInt(args[3]);
            int requiredAttributes = predictorsCount + 1;

            // 1. Cargar datasets
            System.out.println("[1/3] Cargando datasets...");
            Instances trainData = loadDataset(args[0], "Datos de entrenamiento");
            Instances testData = loadDataset(args[1], "Datos de prueba");

            // 2. Normalizar estructura
            System.out.println("\n[2/3] Normalizando estructura...");
            Instances normalizedTest = normalizeStructure(testData, trainData, predictorsCount, requiredAttributes);

            // 3. Guardar resultado
            System.out.println("\n[3/3] Guardando archivo normalizado...");
            saveDataset(normalizedTest, args[2]);

            printSuccessMessage(normalizedTest, trainData.relationName(), predictorsCount);

        } catch (Exception e) {
            handleError(e);
        }
    }

    private static void validateArguments(String[] args) {
        if (args.length != 4) {
            System.err.println("Uso: java ModificarArff <train.arff> <test_blind.arff> <output.arff> <num_predictors>");
            System.err.println("Ejemplo: java ModificarArff train.arff test_blind.arff test_normalized.arff 500");
            System.exit(1);
        }
    }

    private static Instances loadDataset(String path, String datasetName) throws Exception {
        System.out.println("  • " + datasetName + ": " + path);
        Instances data = new DataSource(path).getDataSet();
        if (data.classIndex() < 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    private static Instances normalizeStructure(Instances testData, Instances trainData, int predictorsCount, int requiredAttributes) throws Exception {
        Map<String, Integer> testAttrMap = createAttributeMap(testData);

        ArrayList<Attribute> newAttributes = new ArrayList<>();
        List<Integer> attrMapping = new ArrayList<>();

        addPredictors(trainData, testAttrMap, newAttributes, attrMapping, predictorsCount);
        addClassAttribute(testData, trainData, testAttrMap, newAttributes, attrMapping);

        return createNormalizedDataset(trainData.relationName(), newAttributes, testData, attrMapping, requiredAttributes, predictorsCount);
    }

    private static Map<String, Integer> createAttributeMap(Instances data) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            String name = data.attribute(i).name().toLowerCase();
            if (!name.equals("id")) {
                map.put(name, i);
            }
        }
        return map;
    }

    private static void addPredictors(Instances trainData, Map<String, Integer> testAttrMap,
                                      ArrayList<Attribute> newAttributes, List<Integer> attrMapping, int predictorsCount) {
        int predictorsAdded = 0;

        for (int i = 0; i < trainData.numAttributes() && predictorsAdded < predictorsCount; i++) {
            String name = trainData.attribute(i).name().toLowerCase();

            if (!name.equals("id") && !name.equals("@@class@@")) {
                Attribute attr = trainData.attribute(i);
                String searchName = attr.name().toLowerCase();

                if (testAttrMap.containsKey(searchName)) {
                    attrMapping.add(testAttrMap.get(searchName));
                } else {
                    attrMapping.add(-1);
                }

                newAttributes.add((Attribute) attr.copy());
                predictorsAdded++;
            }
        }

        while (predictorsAdded < predictorsCount) {
            newAttributes.add(new Attribute("ficticio_" + (predictorsAdded + 1)));
            attrMapping.add(-1);
            predictorsAdded++;
        }
    }

    private static void addClassAttribute(Instances testData, Instances trainData,
                                          Map<String, Integer> testAttrMap,
                                          ArrayList<Attribute> newAttributes,
                                          List<Integer> attrMapping) {
        String classAttrName = "@@class@@";
        int testClassIndex = testAttrMap.getOrDefault(classAttrName.toLowerCase(), -1);

        if (testClassIndex != -1) {
            newAttributes.add((Attribute) testData.attribute(testClassIndex).copy());
            attrMapping.add(testClassIndex);
        } else {
            Attribute classAttr = trainData.attribute(trainData.classIndex());
            newAttributes.add((Attribute) classAttr.copy());
            attrMapping.add(-1);
        }
    }

    private static Instances createNormalizedDataset(String relationName,
                                                     ArrayList<Attribute> attributes,
                                                     Instances testData,
                                                     List<Integer> attrMapping,
                                                     int requiredAttributes,
                                                     int predictorsCount) {
        Instances normalized = new Instances(relationName, attributes, testData.numInstances());
        normalized.setClassIndex(predictorsCount);

        for (int i = 0; i < testData.numInstances(); i++) {
            Instance original = testData.instance(i);
            Instance newInst = new DenseInstance(requiredAttributes);

            for (int j = 0; j < requiredAttributes; j++) {
                int originalPos = attrMapping.get(j);
                if (originalPos != -1) {
                    newInst.setValue(j, original.value(originalPos));
                } else {
                    newInst.setMissing(j);
                }
            }
            normalized.add(newInst);
        }
        return normalized;
    }

    private static void saveDataset(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }

    private static void printSuccessMessage(Instances normalizedData, String relationName, int predictorsCount) {
        System.out.println("\n✔ Normalización completada exitosamente");
        System.out.println("════════════════════════════════════════");
        System.out.println("• Archivo de salida creado correctamente");
        System.out.println("• Relación: " + relationName);
        System.out.println("• Atributos totales: " + normalizedData.numAttributes());
        System.out.println("• Predictores: " + predictorsCount);
        System.out.println("• Posición clase: " + normalizedData.classIndex());
        System.out.println("• Instancias procesadas: " + normalizedData.numInstances());
        System.out.println("════════════════════════════════════════");
    }

    private static void handleError(Exception e) {
        System.err.println("\n❌ Error durante la normalización:");
        System.err.println("════════════════════════════════════════");
        System.err.println("Mensaje: " + e.getMessage());
        System.err.println("════════════════════════════════════════");
        e.printStackTrace();
        System.exit(1);
    }
}
