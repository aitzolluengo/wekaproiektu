package preprocessing;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class TransformData {

    /**
     * Aplica StringToWordVector en un dataset ARFF.
     * Convierte texto en valores numéricos para su uso en modelos de Machine Learning.
     *
     * Parámetros:
     * 1. rawData.arff → Archivo de entrada (texto en bruto)
     * 2. dictionary.txt → Archivo donde se guardará el diccionario de palabras
     * 3. transformed.arff → Archivo de salida con los datos procesados
     * Opcionales:
     * -I / --tfidf → Activa TF-IDF
     * -N / --nonsparse → Convierte a formato no disperso (opcional)
     *
     * @param args los argumentos
     * @throws Exception si ocurre un error en WEKA
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 3 || args.length > 5) {
            System.out.println("Uso: java -jar TransformData.jar rawData.arff dictionary.txt transformed.arff [-I/--tfidf] [-N/--nonsparse]");
            return;
        }

        // Leer el dataset ARFF de entrada
        DataSource dataSource = new DataSource(args[0]);
        Instances data = dataSource.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // La clase es el último atributo

        // Configurar el filtro StringToWordVector
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(data);
        filter.setLowerCaseTokens(true);
        filter.setTFTransform(true);  // Activa TF-IDF por defecto
        filter.setIDFTransform(true);
        filter.setDictionaryFileToSaveTo(new File(args[1])); // Guarda el diccionario
        data = Filter.useFilter(data, filter);

        // Verificar si se debe convertir a formato no disperso (Non-Sparse)
        List<String> list = Arrays.asList(args);
        if (list.contains("-N") || list.contains("--nonsparse")) {
            SparseToNonSparse filter2 = new SparseToNonSparse();
            filter2.setInputFormat(data);
            data = Filter.useFilter(data, filter2);
        }

        // Reordenar atributos (pone la clase en la última posición)
        Reorder reorder = new Reorder();
        reorder.setAttributeIndices("2-last,1");
        reorder.setInputFormat(data);
        data = Filter.useFilter(data, reorder);

        // Guardar el dataset transformado
        String outputFile = args[2];  // El archivo de salida siempre será el tercer argumento
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputFile));
        saver.writeBatch();

        System.out.println("Transformación completada. Archivo guardado en: " + outputFile);
    }
}
