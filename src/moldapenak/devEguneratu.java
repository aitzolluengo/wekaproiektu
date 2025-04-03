package moldapenak;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import java.io.*;
import java.util.*;

public class devEguneratu {
    public static void main(String[] args) {
        try {
            validateArguments(args);
            int predictorsCount = Integer.parseInt(args[3]);
            int requiredAttributes = predictorsCount + 1;

            // 1. Kargatu inputak
            System.out.println("[1/3] Cargando datasets...");
            Instances trainData = loadDataset(args[0], "Datos de entrenamiento");
            Instances devData = loadDataset(args[1], "Datos de desarrollo");

            // 2. Estruktura normalizatu
            System.out.println("\n[2/3] Normalizando estructura...");
            Instances normalizedDev = normalizeStructure(devData, trainData, predictorsCount, requiredAttributes);

            // 3. Gorde .arff finala
            System.out.println("\n[3/3] Guardando archivo normalizado...");
            saveDataset(normalizedDev, args[2]);

            printSuccessMessage(normalizedDev, trainData.relationName(), predictorsCount);

        } catch (Exception e) {
            handleError(e);
        }
    }

    // Argumentuak balidatzeko metodoa
    private static void validateArguments(String[] args) {
        if (args.length != 4) {
            System.err.println("Erabilpena: java ModificarArff <train.arff> <dev.arff> <output.arff> <num_predictors>");
            System.err.println("Adibidea: java ModificarArff train.arff dev.arff dev_normalized.arff 500");
            System.exit(1);
        }
    }

    // Datuak kargatzeko metodoa
    private static Instances loadDataset(String path, String datasetName) throws Exception {
        System.out.println("  • " + datasetName + ": " + path);
        Instances data = new DataSource(path).getDataSet();
        if (data.classIndex() < 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    // Egitura normalizatzeko metodoa
    private static Instances normalizeStructure(Instances devData, Instances trainData, int predictorsCount, int requiredAttributes) throws Exception {
        Map<String, Integer> devAttrMap = createAttributeMap(devData);

        ArrayList<Attribute> newAttributes = new ArrayList<>();
        List<Integer> attrMapping = new ArrayList<>();

        // Orden berdinean egoteko balio du for hau
        for (int i = 0; i < trainData.numAttributes(); i++) {
            String name = trainData.attribute(i).name().toLowerCase();
            if (devAttrMap.containsKey(name)) {
                attrMapping.add(devAttrMap.get(name));
            } else {
                attrMapping.add(-1);
            }
            newAttributes.add((Attribute) trainData.attribute(i).copy());
        }

        return createNormalizedDataset(trainData, newAttributes, devData, attrMapping);
    }

    // Atributu mapa bidez sortzeko metodoa
    private static Map<String, Integer> createAttributeMap(Instances data) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            String name = data.attribute(i).name().toLowerCase();
            map.put(name, i);
        }
        return map;
    }

    // Dataset normalizatua sortzeko metodoa
    private static Instances createNormalizedDataset(Instances trainData,
                                                     ArrayList<Attribute> attributes,
                                                     Instances devData,
                                                     List<Integer> attrMapping) {
        Instances normalized = new Instances(trainData.relationName(), attributes, devData.numInstances());
        normalized.setClassIndex(trainData.classIndex());

        for (int i = 0; i < devData.numInstances(); i++) {
            Instance original = devData.instance(i);
            Instance newInst = new DenseInstance(attributes.size());

            for (int j = 0; j < attributes.size(); j++) {
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

    // Datuak gordetzeko metodoa
    private static void saveDataset(Instances data, String outputPath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }

    // Arrakasta mezua inprimatzeko metodoa
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

    // Errorea kudeatzeko metodoa
    private static void handleError(Exception e) {
        System.err.println("\n❌ Error normalizazioan");
        System.err.println("════════════════════════════════════════");
        System.err.println("Mensage: " + e.getMessage());
        System.err.println("════════════════════════════════════════");
        e.printStackTrace();
        System.exit(1);
    }
}