package Sailkatzailea;

import java.io.File;
import java.io.PrintWriter;
import weka.attributeSelection.InfoGainAttributeEval;
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
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();
            if (args.length == 4 && Integer.parseInt(args[3]) <= 350) {
                ranker.setNumToSelect(Integer.parseInt(args[3]));
            }
            ranker.setThreshold(-1.7976931348623157E308);

            AttributeSelection as = new AttributeSelection();
            as.setInputFormat(data);
            as.setEvaluator(evaluator);
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
                pw.println(s); // Guardar el nombre completo del atributo sin modificarlo
            }
            pw.close();
        } else {
            System.out.println("3 parametro behar dira eta zuk " + args.length + " jarri dituzu!!");
            System.out.println("java -jar AtributuenAukeraketa.jar trainPath.arff filteredPath.arff hiztegia atributuLimitea(aukerazkoa)");
        }
    }
}
