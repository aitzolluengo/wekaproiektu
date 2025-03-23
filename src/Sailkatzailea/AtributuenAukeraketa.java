package Sailkatzailea;

import java.io.File;
import java.io.PrintWriter;

import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
//import weka.attributeSelection.AttributeSelection;
import weka.filters.supervised.attribute.AttributeSelection;


// TODO: Auto-generated Javadoc
/**
 * The Class AtributuenAukeraketa.
 */
public class AtributuenAukeraketa {

    /**
     * The main method.
     *
     * @param args the arguments
     * @throws Exception the exception
     */
    public static void main(String[] args) throws Exception{
        if (args.length==3 || args.length==4) {
            DataSource sauce = new DataSource(args[0]); // train data emen sartu
            Instances data = sauce.getDataSet();
            data.setClassIndex(data.numAttributes()-1);


            GainRatioAttributeEval gr = new GainRatioAttributeEval();	//InfoGainRatio atributu ebaluatzailea

            Ranker ranker = new Ranker();		//atributuak rankeatzeko
            if (args.length==4 && Integer.parseInt(args[3])<=300) ranker.setNumToSelect(Integer.parseInt(args[3]));			//atributu kopurua limitatzeko
            ranker.setThreshold(0.1);

            AttributeSelection as = new AttributeSelection();
            as.setInputFormat(data);
            as.setEvaluator(gr);
            as.setSearch(ranker);
            Instances filteredData = Filter.useFilter(data, as);

            ArffSaver saver = new ArffSaver();
            saver.setInstances(filteredData);
            saver.setFile(new File(args[1]));
            saver.writeBatch();

            PrintWriter pw = new PrintWriter(args[2]);				//hiztegia gordetzeko
            pw.println();
            for (int i=0; i<filteredData.numAttributes()-1; i++) {
                String s = filteredData.attribute(i).name();
                pw.println(s.substring(0, s.length() - 2));
            }
            pw.close();

        }
        else {
            System.out.println("3 parametro behar dira eta zuk "+args.length+" jarri dituzu!!");
            System.out.println("java -jar FSS.jar trainPath.arff filteredPath.arff hiztegia atributuLimitea(aukerazkoa)");
        }

    }

}
