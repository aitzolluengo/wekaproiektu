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

public class GetArffBOW {

    /**
     * StringToWordVector aplikatzen du ARFF dataset batean.
     * Testua balio numeriko bihurtzen du Machine Learning ereduetan erabiltzeko.
     *
     * Parametroak:
     * 1. rawData.arff → Sarrerako fitxategia (testu gordina)
     * 2. dictionary.txt → Hitz-hiztegia gordeko den fitxategia
     * 3. transformed.arff → Irteerako fitxategia, prozesatutako datuekin
     * Aukerakoak:
     * -I / --tfidf → TF-IDF aktibatzen du
     * -N / --nonsparse → Ez-disperso formatura bihurtzen du (aukerakoa)
     *
     * @param args argumentuak
     * @throws Exception WEKA-n errore bat gertatzen bada
     */
    public static void main(String[] args) throws Exception {
        try {
            if (args.length < 3 || args.length > 5) {
                System.out.println("Erabilera: java -jar TransformData.jar rawData.arff dictionary.txt transformed.arff [-I/--tfidf] [-N/--nonsparse]");
                return;
            }

            List<String> list = Arrays.asList(args);
            boolean tfidf = list.contains("-I") || list.contains("--tfidf");

            // Sarrerako ARFF dataset-a irakurri
            DataSource dataSource = new DataSource(args[0]);
            Instances data = dataSource.getDataSet();
            if (data.numInstances() == 0) {
                throw new Exception("Datu multzoa hutsik dago kargatu ondoren.");
            }
            data.setClassIndex(data.numAttributes() - 1); // Klasea azken atributua da

            // StringToWordVector iragazkia konfiguratu
            StringToWordVector filter = new StringToWordVector();
            filter.setInputFormat(data);
            filter.setLowerCaseTokens(true);
            filter.setTFTransform(tfidf);
            filter.setIDFTransform(tfidf);
            filter.setDictionaryFileToSaveTo(new File(args[1]));
            data = Filter.useFilter(data, filter);

            // Ez-disperso formatua bihurtu behar den egiaztatu
            if (list.contains("-N") || list.contains("--nonsparse")) {
                SparseToNonSparse filter2 = new SparseToNonSparse();
                filter2.setInputFormat(data);
                data = Filter.useFilter(data, filter2);
            }

            // Atributuak berrantolatu (klasea azken posizioan jartzen du)
            if (data.numAttributes() > 1) {
                Reorder reorder = new Reorder();
                reorder.setAttributeIndices("2-last,1");
                reorder.setInputFormat(data);
                data = Filter.useFilter(data, reorder);
            } else {
                System.out.println("⚠️ Ez da berrantolaketarik aplikatzen: atributu bakarra dago.");
            }

            // Karaktere bereziak dituzten atributu izenak zuzendu
            for (int i = 0; i < data.numAttributes(); i++) {
                String attributeName = data.attribute(i).name();
                if (attributeName.matches(".*[^a-zA-Z0-9_].*")) {
                    data.renameAttribute(i, "'" + attributeName + "'");
                }
            }

            // Eraldatutako datu-multzoa gorde
            String outputFile = args[2];
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(outputFile));
            saver.writeBatch();

            System.out.println("✅ Eraldaketa burututa. Fitxategia hemen gorde da: " + outputFile);
        } catch (Exception e) {
            System.err.println("❌ Errorea eraldaketan: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
