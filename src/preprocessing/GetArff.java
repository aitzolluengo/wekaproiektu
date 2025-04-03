package preprocessing;

import java.io.*;
import java.util.Scanner;

public class GetArff {

    /**
     * ARFF fitxategi bat sortzen du sarrerako direktorio batetik.
     * Direktorioak azpikarpetak baditu, TextDirectoryLoader erabiliko da.
     * Bestela, test_blind modua erabiltzen da.
     *
     * @param args [directorio_entrada] [salida.arff]
     * @throws IOException errore bat gertatzen bada fitxategiak irakurtzean edo idaztean
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Erabilera: java -jar Preprocess.jar [sarrera_direktorioa] [irteera.arff]");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        File inputDir = new File(inputPath);

        if (!inputDir.exists() || !inputDir.isDirectory()) {
            System.out.println("Errorea: bidea " + inputPath + " ez da existitzen edo ez da direktorio bat.");
            return;
        }

        // Azpikarpetak dauden egiaztatu (TextDirectoryLoader erabiltzeko)
        File[] subdirs = inputDir.listFiles(File::isDirectory);

        if (subdirs != null && subdirs.length > 0) {
            convertWithTextDirectoryLoader(inputDir, outputPath);
        } else {
            convertFlatDirectoryToArff(inputDir, outputPath);  // test_blind kasurako
        }
    }

    /**
     * TextDirectoryLoader erabiliz direktorio bat ARFF bihurtzen du.
     *
     * @param dir Sarrerako direktorioa (azpikarpetekin)
     * @param outputPath Irteerako ARFF fitxategia
     * @throws IOException errore bat gertatzen bada
     */
    public static void convertWithTextDirectoryLoader(File dir, String outputPath) throws IOException {
        weka.core.converters.TextDirectoryLoader loader = new weka.core.converters.TextDirectoryLoader();
        loader.setDirectory(dir);
        weka.core.Instances data = loader.getDataSet();

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println(data);
            System.out.println("✅ ARFF sortuta klaseekin (train/dev): " + outputPath);
        }
    }

    /**
     * Azpikarpeta gabeko direktorio bat ARFF bihurtzen du test_blind moduan.
     *
     * @param dir Sarrerako direktorioa
     * @param outputPath Irteerako ARFF fitxategia
     * @throws IOException errore bat gertatzen bada
     */
    public static void convertFlatDirectoryToArff(File dir, String outputPath) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(outputPath));
        pw.println("@relation test_blind");
        pw.println("@attribute text string");
        pw.println("@data");  // @attribute class ez da gehitzen (test_blind ez duelako etiketarik)

        File[] files = dir.listFiles(File::isFile);
        int count = 0;

        for (File file : files) {
            StringBuilder sb = new StringBuilder();
            try (Scanner scanner = new Scanner(file, "UTF-8")) {
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    line = line.replace("'", "\\'");  // Komatxoak ihes egitea
                    sb.append(line.trim()).append(" ");
                }
            }

            String content = sb.toString().trim().replaceAll("\\s+", " ");

            if (!content.isEmpty()) {
                pw.println("'" + content + "'");  // Etiketarik gabe
                count++;
            }
        }

        pw.close();
        System.out.println("✅ ARFF sortuta test_blind moduan " + count + " instantziekin → " + outputPath);
    }
}
