package Sailkatzailea;


import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

// TODO: Auto-generated Javadoc
/**
 * The Class ParametroEkorketa.
 */
public class ParametroenBilaketa {

    /** The pw. */
    private static PrintWriter pw;

    /** The data. */
    private static Instances data;


    /**
     * The main method.
     *
     * @param args the arguments
     * @throws Exception the exception
     */
    public static void main(String[] args) throws Exception {

        if (args.length==2) {
            DataSource source = new DataSource(args[0]);
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);
            double maximoa = 0.0;
            FileWriter filewriter = new FileWriter(args[1]);
            pw = new PrintWriter(filewriter);
            pw.println();
            pw.println("SMO parametro ekorketa");
            pw.println("2 parametro optimizatuko ditugu:");
            pw.println("1- C");
            pw.println("2- Kernel.gamma");
            pw.println("Ebaluazio metrika: Klase minoritarioaren fMeasure");
            pw.println();

            SMO smo = new SMO();
            double oinarria=10;
            double cmax=0.0;
            double gammamax=0.0;
            RBFKernel kernel= new RBFKernel();
            int i = klaseminoritarioa();
            for (double exp =-3.0; exp<4.0; exp+=1.0) {
                double c=Math.pow(oinarria,exp);
                smo.setC(c);
                for (double exp1 =-3.0; exp1<4.0; exp1+=1.0) {
                    double gamma=Math.pow(oinarria,exp1);
                    kernel.setGamma(gamma);
                    smo.setKernel(kernel);
                    Evaluation eval= new Evaluation(data);
                    eval.crossValidateModel(smo, data, 10, new Random(1));

                    if (eval.fMeasure(i) > maximoa) {
                        maximoa = eval.fMeasure(i);
                        cmax=c;
                        gammamax=gamma;
                    }
                }
            }
            pw.println("C hoberena:");
            pw.println(cmax);
            pw.println("Gamma hoberena:");
            pw.println(gammamax);
            pw.println("Klase minoritarioaren fMeasure hoberena:");
            pw.println(maximoa);
            pw.close();
        }
        else {
            System.out.println(args.length + " parametro sartu dituzu");
            System.out.println("2 parametro sartu behar dituzu!");
            System.out.println("java -jar ParametroEkorketa.jar data.arff parametroekorketa.txt");
        }
    }

    /**
     * Klaseminoritarioa.
     *
     * @return the int
     * @throws Exception the exception
     */
    private static int klaseminoritarioa() throws Exception{

        //Klase minoritarioaren indizea itzultzen du
        pw.println(data.attribute(data.numAttributes()-1).name()+" atributu nominala da eta hauek dira ezaugarriak:");
        int[] counts = data.attributeStats(data.numAttributes()-1).nominalCounts;
        int min = 0; //Hemen klase minoritarioaren posizioa gordeko da
        for(int j=0; j<counts.length; j++){
            if(counts[min] > counts[j]) {
                min = j;
            }
            pw.println(data.attribute(data.numAttributes()-1).value(j) + " -> " + counts[j] + " | Maiztasuna -> " + (float)counts[j]/data.attributeStats(data.numAttributes()-1).totalCount);
        }
        pw.println("Balio minimoa: " + data.attribute(data.numAttributes()-1).value(min) + " -> " + counts[min] + "\n");
        return min;
    }

}
