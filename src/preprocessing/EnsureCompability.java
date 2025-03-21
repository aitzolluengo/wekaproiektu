package preprocessing;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Reorder;

// TODO: Auto-generated Javadoc
/**
 * The Class MakeCompatible.
 */

//ATRIBUTUEN HAUTAPENA//

public class EnsureCompability {

    /**
     * The main method.
     *
     * @param args the arguments
     * @throws Exception the exception
     */
    public static void main(String[] args) throws Exception {

        if(args.length  !=3) {
            System.out.println("Ez duzu arguments atala behar bezala bete!");
            System.out.println(args.length + " parametro sartu dituzu");
            System.out.println("java -jar test.arff hiztegia.txt emaitza.arff");
        }
        else{
            String arffIn= args[0];
            String dictionary= args[1];
            String arffOut= args[2];
            for(String x: args)
                System.out.println(x);

            //dev multzoa lortu (arguments atalean sartzen den lehenengo parametroa args[0])
            DataSource source=null;
            if (arffIn.contains(".csv")) convertCSVtoArff(arffIn);

            try {
                source = new DataSource(arffIn);
            } catch (Exception e) {
                System.out.println("ERROREA - Sarrerako fitxategiaren helbidea okerra da");
                return; // Salir del programa si hay un error
            }
            Instances dev= source.getDataSet();

            //dev multzoaren klasea definitu
            dev.setClassIndex(dev.numAttributes()-1);

            //Orain parametro bezala (dictionary) lortu dugun hiztegia dev ean sartuko dugu FixedDictionaryStringToWordVector erabiliz
            FixedDictionaryStringToWordVector hiztegia= new FixedDictionaryStringToWordVector();
            hiztegia.setDictionaryFile(new File(dictionary));
            hiztegia.setInputFormat(dev);
            dev=Filter.useFilter(dev, hiztegia);

            // Corregir nombres de atributos con caracteres especiales
            for (int i = 0; i < dev.numAttributes(); i++) {
                String attributeName = dev.attribute(i).name();
                // Si el nombre del atributo contiene caracteres especiales, escápalo con comillas simples
                if (attributeName.matches(".*[^a-zA-Z0-9_].*")) {
                    dev.renameAttribute(i, "'" + attributeName + "'");
                }
            }

            //Atributuak reordenartu, klasea amaieran agertu dadin, horretarako reorder filtroa erabiliko dugu
            Reorder reorder = new Reorder();
            reorder.setAttributeIndices("2-" + dev.numAttributes() + ",1");
            reorder.setInputFormat(dev);
            dev = Filter.useFilter(dev, reorder);

            //arff fitxategia berria sortu arguments atalean sartu dugun helbidean (arffOut)
            ArffSaver arffSaver = new ArffSaver();
            arffSaver.setInstances(dev);
            arffSaver.setFile(new File(arffOut));
            arffSaver.writeBatch();

            System.out.println("Transformación completada. Archivo guardado en: " + arffOut);
        }
    }

    /**
     * Convert CS vto arff.
     *
     * @param filename the filename
     * @throws Exception the exception
     */
    public static void convertCSVtoArff(String filename) throws Exception {

        // CSV-a kargatu
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filename));

        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);

        filename = filename.replace(".csv", ".arff");

        saver.setFile(new File(filename));
        saver.writeBatch();
    }
}