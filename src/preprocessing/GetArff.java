package preprocessing;

import java.io.*;
import java.util.Scanner;

public class Preprocess {

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Uso: java -jar Preprocess.jar [directorio_entrada] [salida.arff]");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        File inputDir = new File(inputPath);

        if (!inputDir.exists() || !inputDir.isDirectory()) {
            System.out.println("Error: la ruta " + inputPath + " no existe o no es un directorio.");
            return;
        }

        // Verificamos si hay subcarpetas (para usar TextDirectoryLoader)
        File[] subdirs = inputDir.listFiles(File::isDirectory);

        if (subdirs != null && subdirs.length > 0) {
            convertWithTextDirectoryLoader(inputDir, outputPath);
        } else {
            convertFlatDirectoryToArff(inputDir, outputPath);  // para test_blind
        }
    }

    public static void convertWithTextDirectoryLoader(File dir, String outputPath) throws IOException {
        weka.core.converters.TextDirectoryLoader loader = new weka.core.converters.TextDirectoryLoader();
        loader.setDirectory(dir);
        weka.core.Instances data = loader.getDataSet();

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println(data);
            System.out.println("✅ ARFF generado con clases (train/dev): " + outputPath);
        }
    }

    public static void convertFlatDirectoryToArff(File dir, String outputPath) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(outputPath));
        pw.println("@relation test_blind");
        pw.println("@attribute text string");
        pw.println("@data");  // Sin @attribute class (porque test_blind no tiene etiquetas)

        File[] files = dir.listFiles(File::isFile);
        int count = 0;

        for (File file : files) {
            StringBuilder sb = new StringBuilder();
            try (Scanner scanner = new Scanner(file, "UTF-8")) {
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    line = line.replace("'", "\\'");  // Escapar comillas
                    sb.append(line.trim()).append(" ");
                }
            }

            String content = sb.toString().trim().replaceAll("\\s+", " ");

            if (!content.isEmpty()) {
                pw.println("'" + content + "'");  // Sin etiqueta
                count++;
            }
        }

        pw.close();
        System.out.println("✅ ARFF generado para test_blind con " + count + " instancias → " + outputPath);
    }
}
