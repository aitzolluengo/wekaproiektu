package Sailkatzailea;

import java.io.File;
import java.io.PrintWriter;

import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class AtributuenAukeraketa {

    public static void main(String[] args) throws Exception {
        if (args.length == 3 || args.length == 4) {
            // Cargar datos
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Aplicar selección de atributos
            GainRatioAttributeEval gr = new GainRatioAttributeEval();
            Ranker ranker = new Ranker();
            if (args.length == 4 && Integer.parseInt(args[3]) <= 300) {
                ranker.setNumToSelect(Integer.parseInt(args[3]));
            }
            ranker.setThreshold(0.1);

            AttributeSelection as = new AttributeSelection();
            as.setInputFormat(data);
            as.setEvaluator(gr);
            as.setSearch(ranker);
            Instances filteredData = Filter.useFilter(data, as);

            // Guardar el conjunto de datos filtrado
            ArffSaver saver = new ArffSaver();
            saver.setInstances(filteredData);
            saver.setFile(new File(args[1]));
            saver.writeBatch();

            // Guardar la lista de atributos seleccionados
            PrintWriter pw = new PrintWriter(args[2]);
            pw.println();
            for (int i = 0; i < filteredData.numAttributes() - 1; i++) {
                String s = filteredData.attribute(i).name();
                // Verificar que el nombre del atributo tenga al menos 2 caracteres
                if (s.length() >= 2) {
                    pw.println(s.substring(0, s.length() - 2));
                } else {
                    pw.println(s); // Si es demasiado corto, imprimir el nombre completo
                }
            }
            pw.close();
        } else {
            System.out.println("3 parametro behar dira eta zuk " + args.length + " jarri dituzu!!");
            System.out.println("java -jar AtributuenAukeraketa.jar trainPath.arff filteredPath.arff hiztegia atributuLimitea(aukerazkoa)");
        }
    }
}