package Sailkatzailea;

import weka.classifiers.Evaluation;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.global.HillClimber;
import weka.classifiers.bayes.net.search.global.K2;
import weka.classifiers.bayes.net.search.global.TabuSearch;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;

public class ParametroenBilaketa {
    private static PrintWriter pw;

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println(args.length + " parametro sartu dituzu");
            System.out.println("3 parametro sartu behar dituzu!");
            System.out.println("java -jar ParametroenBilaketa.jar data.arff parametroak.txt Bayes Network|Naive Bayes");
            return;
        }

        String algoritmo = args[2];
        DataSource source = new DataSource(args[0]); // Datu sorta kargatu
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // Klasearen indizea ezarri
        pw = new PrintWriter(new FileWriter(args[1]));

        data.randomize(new java.util.Random(1));
        
        //Hold out egin
        RemovePercentage RP = new RemovePercentage();
        RP.setPercentage(70);
        RP.setInvertSelection(true);
        RP.setInputFormat(data);
        Instances train = Filter.useFilter(data, RP); // Train datu sorta

        RP.setInvertSelection(false);
        Instances test = Filter.useFilter(data, RP); // Dev datu sorta

        int i = klaseminoritarioa(data);

        // Zein algoritmo hautatu den jakiteko logika
        if (algoritmo.equalsIgnoreCase("Bayes Network")) {
            optimizeBayesNetwork(train, test);
        } else if (algoritmo.equalsIgnoreCase("Naive Bayes")) {
            optimizeNaiveBayes(train, test, i);
        } else {
            System.out.println("Algoritmo okerra. Erabili 'Bayes Network' edo 'Naive Bayes'.");
        }

        pw.flush();
        pw.close();
    }

    private static void optimizeBayesNetwork(Instances train, Instances test) throws Exception {
        
        // Bilaketa algoritmo optimoena eta estimatzailearen alpha balio optimoa bilatuko ditugu
        SearchAlgorithm[] searchAlgorithms = {new K2(), new HillClimber(), new TabuSearch()};
        double[] alphaValues = {0.1, 0.5, 1.0};

        double maxAccuracy = 0.0;
        SearchAlgorithm bestSearchAlgorithm = null;
        double bestAlpha = 0.1;
        //Parametroak bilatzen
        for (SearchAlgorithm searchAlgorithm : searchAlgorithms) {
            for (double alpha : alphaValues) {
                BayesNet bnTemp = new BayesNet();
                bnTemp.setSearchAlgorithm(searchAlgorithm); // Bilaketa algoritmoa hemen frogatuko da

                SimpleEstimator estimator = new SimpleEstimator(); //Estimatzailea beti berdina izango da
                estimator.setAlpha(alpha); //Alpha balioak hemen frogatuko dira
                bnTemp.setEstimator(estimator);

                bnTemp.buildClassifier(train); 
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(bnTemp, test); //Ebaluazioa f-measure lortzeko

                // F-measure handiena lortzen duten parametroak gordeko dira
                if (eval.fMeasure(1) > maxAccuracy) {
                    maxAccuracy = eval.fMeasure(1);
                    bestSearchAlgorithm = searchAlgorithm;
                    bestAlpha = alpha;
                }
            }
        }

        String result = "Best SearchAlgorithm: " + bestSearchAlgorithm.getClass().getSimpleName() +
                "\nBest Alpha: " + bestAlpha +
                "\nMax Accuracy: " + maxAccuracy;
        System.out.println(result);
        pw.println(result);
    }

    private static void optimizeNaiveBayes(Instances train, Instances test, int i) throws Exception {
        //Hurrengo bi aukeratan True edo False artean zein den hoberena ikusiko dugu 
        boolean[] kernelEstimatorOptions = {true, false};
        boolean[] discretizationOptions = {true, false};

        double maxAccuracy = 0.0;
        boolean bestKernelEstimator = false;
        boolean bestDiscretization = false;
        //Parametro optimoak lortzen
        for (boolean useKernelEstimator : kernelEstimatorOptions) {
            for (boolean useDiscretization : discretizationOptions) {
                // Errorea ekiditzeko, ezin dira biak aldi berean eman
                if (useKernelEstimator && useDiscretization) {
                    continue;  
                }

                NaiveBayes nb = new NaiveBayes();
                nb.setUseKernelEstimator(useKernelEstimator); //Balioak frogatzen
                nb.setUseSupervisedDiscretization(useDiscretization); //Balioak frogatzen

                nb.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(nb, test); //Ebaluazioa f-measure lortzeko
                // F-measure handiena lortzen duten parametroak gordeko dira
                if (eval.fMeasure(i) > maxAccuracy) {
                    maxAccuracy = eval.fMeasure(i);
                    bestKernelEstimator = useKernelEstimator;
                    bestDiscretization = useDiscretization;
                }
            }
        }

        String result = "Best KernelEstimator: " + bestKernelEstimator +
                "\nBest Discretization: " + bestDiscretization +
                "\nMax Accuracy: " + maxAccuracy;
        System.out.println(result);
        pw.println(result);
    }
    // Klase minoritarioa lortzeko, izan ere, klase minoritarioaren F-measure lortu behar dugu parametro ekorketan
    private static int klaseminoritarioa(Instances data) {
        int minfreq = Integer.MAX_VALUE;
        int minclassindex = 0;

        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            int freq = data.attributeStats(data.classIndex()).nominalCounts[i]; //Klase bakoitza zenbatetan agertzen den zenbatu
            if (freq < minfreq) {
                minfreq = freq;
                minclassindex = i; // Klase minoritarioa
            }
        }

        String result = "Klase minoritarioa: " + minclassindex + " Maiztasuna: " + minfreq;
        System.out.println(result);
        pw.println(result);

        return minclassindex;
    }
}
