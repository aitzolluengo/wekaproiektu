package Iragarpenak;

import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class ModificarArff {
    public static void main(String[] args) {
        try {
            if (args.length != 4) {
                System.err.println("Uso: java ModificarArff <input.arff> <output.arff> <atributos_modelo.txt> <numAtributos>");
                System.exit(1);
            }

            System.out.println("Cargando datos de prueba...");
            Instances testData = new DataSource(args[0]).getDataSet();
            System.out.printf("Atributos en test data: %d%n", testData.numAttributes());

            System.out.println("Procesando atributos del modelo...");
            List<String> nombresAtributos = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(args[2]))) {
                String line;
                boolean inMetadata = true;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.startsWith("===")) {
                        inMetadata = !inMetadata;
                        continue;
                    }
                    if (!inMetadata && !line.isEmpty() && line.contains("|")) {
                        String[] parts = line.split("\\|", 2);
                        if (parts.length == 2 && !parts[1].trim().isEmpty()) {
                            nombresAtributos.add(parts[1].trim());
                        }
                    }
                }
            }
            System.out.printf("Atributos válidos encontrados: %d%n", nombresAtributos.size());

            // Crear mapa de nombres de atributos del test (normalizados)
            Map<String, Integer> testAttrMap = new HashMap<>();
            for (int i = 0; i < testData.numAttributes(); i++) {
                String nombre = testData.attribute(i).name().replaceAll("['\"]", "").toLowerCase();
                testAttrMap.put(nombre, i + 1); // +1 para Weka
            }

            List<Integer> indicesSeleccionados = new ArrayList<>();
            for (String nombreModelo : nombresAtributos) {
                String nombreNormalizado = nombreModelo.replaceAll("['\"]", "").toLowerCase();
                if (testAttrMap.containsKey(nombreNormalizado)) {
                    indicesSeleccionados.add(testAttrMap.get(nombreNormalizado));
                } else {
                    System.out.println("⚠ No encontrado: " + nombreModelo);
                }
            }

            int numRequerido = Integer.parseInt(args[3]);
            if (indicesSeleccionados.size() < numRequerido) {
                System.err.printf("Advertencia: Solo se encontraron %d de %d atributos requeridos%n",
                        indicesSeleccionados.size(), numRequerido);
                numRequerido = indicesSeleccionados.size();
            }

            String indicesStr = indicesSeleccionados.stream()
                    .limit(numRequerido)
                    .map(Object::toString)
                    .collect(Collectors.joining(","));

            Reorder reorder = new Reorder();
            reorder.setAttributeIndices(indicesStr);
            reorder.setInputFormat(testData);
            Instances filteredData = Filter.useFilter(testData, reorder);

            Attribute clase = new Attribute("'@@class@@'", Arrays.asList("pos", "neg"));
            filteredData.insertAttributeAt(clase, filteredData.numAttributes());
            filteredData.setClassIndex(filteredData.numAttributes() - 1);

            for (Instance instancia : filteredData) {
                instancia.setMissing(filteredData.classIndex());
            }

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(args[1]))) {
                writer.write(filteredData.toString());
            }

            System.out.println("Proceso completado con éxito!");
            System.out.printf("Atributos en output: %d%n", filteredData.numAttributes());

        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
 }
}
}
