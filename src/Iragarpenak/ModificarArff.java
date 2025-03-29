package Iragarpenak;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import java.io.*;
import java.util.*;

public class ModificarArff {
    private static final int REQUIRED_ATTRIBUTES = 1001; // 1000 predictores + 1 clase
    private static final int PREDICTORS_COUNT = 1000;

    public static void main(String[] args) {
        try {
            validateArguments(args);

            // 1. Cargar datasets
            System.out.println("[1/3] Cargando datasets...");
            Instances trainData = loadDataset(args[0], "Datos de entrenamiento");
            Instances testData = loadDataset(args[1], "Datos de prueba");

            // 2. Normalizar estructura
            System.out.println("\n[2/3] Normalizando estructura...");
            Instances normalizedTest = normalizeStructure(testData, trainData);

            // 3. Guardar resultado
            System.out.println("\n[3/3] Guardando archivo normalizado...");
            saveDataset(normalizedTest, args[2]);

            printSuccessMessage(normalizedTest, trainData.relationName());

        } catch (Exception e) {
            handleError(e);
        }
    }

    private static void validateArguments(String[] args) {
        if (args.length != 3) {
            System.err.println("Uso: java ArffNormalizer <train.arff> <test_blind.arff> <output.arff>");
            System.err.println("Ejemplo: java ArffNormalizer train.arff test_blind.arff test_normalized.arff");
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

    private static Instances normalizeStructure(Instances testData, Instances trainData) throws Exception {
        // Mapeo de atributos del test
        Map<String, Integer> testAttrMap = createAttributeMap(testData);

        // Construir nueva estructura basada en train
        ArrayList<Attribute> newAttributes = new ArrayList<>();
        List<Integer> attrMapping = new ArrayList<>();

        // 1. Agregar predictores (primeros 1000 atributos de train, excluyendo ID y clase)
        addPredictors(trainData, testAttrMap, newAttributes, attrMapping);

        // 2. Agregar atributo clase
        addClassAttribute(testData, trainData, testAttrMap, newAttributes, attrMapping);

        // Crear dataset normalizado
        Instances normalized = createNormalizedDataset(trainData.relationName(), newAttributes, testData, attrMapping);

        validateNormalizedDataset(normalized);
        return normalized;
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
                                      ArrayList<Attribute> newAttributes, List<Integer> attrMapping) {
        int predictorsAdded = 0;

        for (int i = 0; i < trainData.numAttributes() && predictorsAdded < PREDICTORS_COUNT; i++) {
            String name = trainData.attribute(i).name().toLowerCase();

            if (!name.equals("id") && !name.equals("@@class@@")) {
                Attribute attr = trainData.attribute(i);
                String searchName = attr.name().toLowerCase();

                if (testAttrMap.containsKey(searchName)) {
                    attrMapping.add(testAttrMap.get(searchName));
                } else {
                    attrMapping.add(-1); // Atributo faltante
                }

                newAttributes.add((Attribute) attr.copy());
                predictorsAdded++;
            }
        }

        // Completar con atributos ficticios si es necesario
        while (predictorsAdded < PREDICTORS_COUNT) {
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
            // Crear atributo clase con valores missing si no existe
            Attribute classAttr = trainData.attribute(trainData.classIndex());
            newAttributes.add((Attribute) classAttr.copy());
            attrMapping.add(-1);
        }
    }

    private static Instances createNormalizedDataset(String relationName,
                                                     ArrayList<Attribute> attributes,
                                                     Instances testData,
                                                     List<Integer> attrMapping) {
        Instances normalized = new Instances(relationName, attributes, testData.numInstances());
        normalized.setClassIndex(PREDICTORS_COUNT); // La clase es el último atributo

        for (int i = 0; i < testData.numInstances(); i++) {
            Instance original = testData.instance(i);
            Instance newInst = new DenseInstance(REQUIRED_ATTRIBUTES);

            for (int j = 0; j < REQUIRED_ATTRIBUTES; j++) {
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

    private static void validateNormalizedDataset(Instances data) throws Exception {
        if (data.numAttributes() != REQUIRED_ATTRIBUTES) {
            throw new Exception("Dataset normalizado debe tener exactamente " +
                    REQUIRED_ATTRIBUTES + " atributos");
        }

        if (data.classIndex() != PREDICTORS_COUNT) {
            throw new Exception("Índice de clase incorrecto. Debe ser " + PREDICTORS_COUNT);
        }
    }

    private static void saveDataset(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }

    private static void printSuccessMessage(Instances normalizedData, String relationName) {
        System.out.println("\n✔ Normalización completada exitosamente");
        System.out.println("════════════════════════════════════════");
        System.out.println("• Archivo de salida creado correctamente");
        System.out.println("• Relación: " + relationName);
        System.out.println("• Atributos totales: " + normalizedData.numAttributes());
        System.out.println("• Predictores: " + PREDICTORS_COUNT);
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
