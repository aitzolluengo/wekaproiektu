package preprocessing;

import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

public class Preprocess {

    public static void main(String[] args) {
        try {
            // Verificar que los argumentos sean correctos
            if (args.length != 2) {
                System.out.println("Uso: java -jar Preprocess.jar [input_folder_or_file] [output_file.arff]");
                return;
            }

            // Obtener los parámetros de entrada
            String inputPath = args[0];  // Carpeta o archivo a procesar
            String outputFile = args[1]; // Archivo .arff de salida

            File inputFile = new File(inputPath);

            // Verificar si la entrada es una carpeta o un archivo único
            if (inputFile.isDirectory()) {
                convertDirectoryToArff(inputPath, outputFile);  // Convertir carpeta (SPAM y HAM)
            } else if (inputFile.isFile()) {
                convertTextFileToArff(inputPath, outputFile);   // Convertir un solo archivo `.txt`
            } else {
                System.out.println("Error: La ruta de entrada no existe o no es válida.");
            }
        } catch (Exception e) {
            System.err.println("Error inesperado: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Convierte una carpeta con archivos `.txt` en un `.arff` usando `TextDirectoryLoader`.
     * @param directory Ruta de la carpeta que contiene los archivos (deben estar en subcarpetas "spam" y "ham").
     * @param target Ruta donde se guardará el archivo `.arff` generado.
     * @throws IOException Si ocurre un error al leer/escribir archivos.
     */
    public static void convertDirectoryToArff(String directory, String target) throws IOException {
        File dir = new File(directory);
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IOException("El directorio especificado no existe o no es un directorio válido: " + directory);
        }

        // Crear un cargador de directorios de texto de WEKA
        TextDirectoryLoader loader = new TextDirectoryLoader();
        loader.setDirectory(dir);

        // Convertir los archivos en un conjunto de datos WEKA
        Instances data = loader.getDataSet();

        // Escribir el archivo ARFF generado
        try (PrintWriter writer = new PrintWriter(new FileWriter(target))) {
            writer.println(data);
            System.out.println("Archivo ARFF generado correctamente: " + target);
        } catch (IOException e) {
            System.err.println("Error al escribir el archivo ARFF: " + target);
            throw e;
        }
    }

    /**
     * Convierte un único archivo `.txt` en un `.arff` con formato compatible con WEKA.
     * @param file Ruta del archivo de texto a convertir.
     * @param target Ruta donde se guardará el archivo `.arff` generado.
     * @throws IOException Si ocurre un error al leer/escribir archivos.
     */
    public static void convertTextFileToArff(String file, String target) throws IOException {
        File inputFile = new File(file);
        if (!inputFile.exists() || !inputFile.isFile()) {
            throw new IOException("El archivo especificado no existe o no es un archivo válido: " + file);
        }

        try (PrintWriter pw = new PrintWriter(new FileWriter(target))) {
            // Escribir la cabecera del archivo ARFF
            pw.println("@relation blindSpam\n");
            pw.println("@attribute text string");
            pw.println("@attribute class {spam, ham}\n");
            pw.println("@data");

            // Leer el contenido del archivo y escribirlo en el formato ARFF
            try (Scanner scanner = new Scanner(inputFile)) {
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine().toLowerCase();
                    line = line.replaceAll("'", " ");  // Reemplazar comillas simples para evitar errores
                    pw.println("'" + line + "',?");
                }
            } catch (IOException e) {
                System.err.println("Error al leer el archivo: " + file);
                throw e;
            }
        } catch (IOException e) {
            System.err.println("Error al escribir el archivo ARFF: " + target);
            throw e;
        }

        System.out.println("Archivo ARFF generado para archivo único: " + target);
    }
}