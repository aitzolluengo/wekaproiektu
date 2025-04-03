package Sailkatzailea;

import java.io.*;
import weka.attributeSelection.*;
import weka.core.*;
import weka.core.converters.*;
import weka.filters.*;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ConverterUtils.DataSource;

public class FSS {
    public static void main(String[] args) throws Exception {
        // Parametroen hobekuntzan datza 
        if (args.length < 3 || args.length > 4) {
            System.err.println("Erabilpena : java AtributuenAukeraketa <input.arff> <output.arff> <diccionario.txt> [numAtributos]");
            System.err.println("  numAtributos: Opcional (valor por defecto: todos)");
            System.exit(1);
        }

        try {
            // 1. Datuak kargatu
            System.out.println("Kargatzen dataset...");
            Instances data = new DataSource(args[0]).getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            System.out.printf("Atributoak kargatutak: %d%n", data.numAttributes());

            // 2. Ze atributu hartu kodifikatu
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();

            // Maneiatu parametro kopurua (laugarren argumentua)
            if (args.length == 4) {
                int numAttr = Integer.parseInt(args[3]);
                if (numAttr <= 0 || numAttr > data.numAttributes()) {
                    System.err.println("Número de atributos inválido. Usando todos.");
                } else {
                    ranker.setNumToSelect(numAttr);
                    System.out.printf("Seleccionando los %d mejores atributos%n", numAttr);
                }
            }
            ranker.setThreshold(-1.7976931348623157E308);

            // 3. Filtroak aplikatu
            AttributeSelection filter = new AttributeSelection();
            filter.setEvaluator(evaluator);
            filter.setSearch(ranker);
            filter.setInputFormat(data);
            Instances filteredData = Filter.useFilter(data, filter);
            System.out.printf("Atributos después de filtrar: %d%n", filteredData.numAttributes());

            // 4. Gorde datuak
            ArffSaver saver = new ArffSaver();
            saver.setInstances(filteredData);
            saver.setFile(new File(args[1]));
            saver.writeBatch();

            // 5. Gorde hiztegia metadatuekin
            try (PrintWriter pw = new PrintWriter(args[2])) {
                pw.println("=== METADATOS DE ATRIBUTOS ===");
                pw.printf("Total atributos originales: %d%n", data.numAttributes());
                pw.printf("Atributos seleccionados: %d%n", filteredData.numAttributes() - 1);
                pw.println("Formato: índice_original|nombre_atributo");

                // Gorde hasiera
                for (int i = 0; i < filteredData.numAttributes() - 1; i++) {
                    String attrName = filteredData.attribute(i).name();
                    // Bilatu hasiera
                    int originalIndex = -1;
                    for (int j = 0; j < data.numAttributes(); j++) {
                        if (data.attribute(j).name().equals(attrName)) {
                            originalIndex = j;
                            break;
                        }
                    }
                    pw.printf("%d|%s%n", originalIndex + 1, attrName); // +1 para compatibilidad con Weka
                }
            }

            System.out.println("Proceso completado con éxito!");
            System.out.printf("Diccionario guardado en: %s%n", args[2]);

        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
