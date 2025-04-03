package preprocessing;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;

public class EnsureCompatibility {

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("Uso: java -jar EnsureCompatibility.jar input.arff dictionary.txt output.arff");
            return;
        }

        String inputArff = args[0];
        String dictionaryFile = args[1];
        String outputArff = args[2];

        // Load input dataset
        DataSource source = new DataSource(inputArff);
        Instances data = source.getDataSet();

        // Set class index (last attribute)
        data.setClassIndex(data.numAttributes() - 1);

        // Apply FixedDictionaryStringToWordVector
        FixedDictionaryStringToWordVector fdsv = new FixedDictionaryStringToWordVector();
        fdsv.setDictionaryFile(new File(dictionaryFile));
        fdsv.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, fdsv);

        // Fix: Properly reorder attributes to move class to end
        if (filteredData.classIndex() >= 0) {
            Reorder reorder = new Reorder();

            // Build attribute indices string:
            // 1. All attributes except class
            // 2. Then the class attribute
            String indices = "";

            // Add all attributes before class
            if (filteredData.classIndex() > 0) {
                indices += "1-" + filteredData.classIndex();
            }

            // Add all attributes after class
            if (filteredData.classIndex() < filteredData.numAttributes() - 1) {
                if (!indices.isEmpty()) indices += ",";
                indices += (filteredData.classIndex() + 2) + "-" + filteredData.numAttributes();
            }

            // Finally add the class attribute
            if (!indices.isEmpty()) indices += ",";
            indices += (filteredData.classIndex() + 1);

            reorder.setAttributeIndices(indices);
            reorder.setInputFormat(filteredData);
            filteredData = Filter.useFilter(filteredData, reorder);
        }

        // Save output
        ArffSaver saver = new ArffSaver();
        saver.setInstances(filteredData);
        saver.setFile(new File(outputArff));
        saver.writeBatch();

        System.out.println("Transformación completada. Archivo guardado en: " + outputArff);
    }
}
