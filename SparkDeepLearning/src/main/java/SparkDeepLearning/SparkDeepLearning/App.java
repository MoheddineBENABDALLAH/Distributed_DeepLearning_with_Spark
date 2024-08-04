/*package SparkDeepLearning.SparkDeepLearning;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    private int batchSize = 64;
    private int numEpochs = 500;

    public static void main(String[] args) throws Exception {
        new App().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {

        // Load the data
        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, 12345);

        // Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Execute training
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIter);
            double score = model.score();
            log.info("Completed Epoch {} with score {}", i, score);
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime; // Time taken in milliseconds

        // Save the model
        model.save(new File("/home/cluster/model2.bin"));

        // Load the model for evaluation
        MultiLayerNetwork restoredModel = MultiLayerNetwork.load(new File("/home/cluster/model2.bin"), true);

        // Perform evaluation
        Evaluation eval = restoredModel.evaluate(testIter);

        long startTime2 = System.currentTimeMillis();
        long endTime2 = System.currentTimeMillis();
        long duration2 = endTime2 - startTime2; // Time taken in milliseconds

        log.info("***** Evaluation *****");
        log.info(eval.stats());
        log.info("Training Time: {} milliseconds", duration);
        log.info("Testing Time: {} milliseconds", duration2);

        log.info("***** Example Complete *****");
    }
}
*/

package SparkDeepLearning.SparkDeepLearning;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);


    private int batchSizePerWorker = 64;

    private int numEpochs = 100;

    public static void main(String[] args) throws Exception {
        new App().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {

        SparkConf sparkConf = new SparkConf();

        sparkConf.setMaster("yarn");

        sparkConf.setAppName("DL4J Spark MLP Example");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data into memory then parallelize
        //This isn't a good approach in general - but is simple to use for this example
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, false, 12345);
        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList).persist(StorageLevel.MEMORY_ONLY_SER());
        JavaRDD<DataSet> testData = sc.parallelize(testDataList).persist(StorageLevel.MEMORY_ONLY_SER());


        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
                .build();




        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(10)
                .workerPrefetchNumBatches(10)            //Async prefetching: 2 examples per worker
                .batchSizePerWorker(batchSizePerWorker)
                .build();



        //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        //Execute training:
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < numEpochs; i++) {
        	 sparkNet.fit(trainData);
            double score = sparkNet.getScore();
            log.info("Completed Epoch {} with score {}", i, score);
            //log.info("Completed Epoch {}", i);
        }
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime; // Time taken in milliseconds


        MultiLayerNetwork netApresTrain= sparkNet.getNetwork();
        netApresTrain.save(new File("/home/user/model4.bin"));

        MultiLayerNetwork netForEval= MultiLayerNetwork.load(new File("/home/user/model4.bin"),true);
        //Perform evaluation (distributed)
        SparkDl4jMultiLayer sparkNet2 = new SparkDl4jMultiLayer(sc, netForEval, tm);
        long startTime2 = System.currentTimeMillis();

        Evaluation evaluation = sparkNet2.evaluate(testData);
        long endTime2 = System.currentTimeMillis();
        long duration2 = endTime2 - startTime2; // Time taken in milliseconds

        log.info("***** Evaluation *****");
        log.info(evaluation.stats());
        log.info("Training Time: {} milliseconds", duration);
        log.info("Testing Time: {} milliseconds", duration2);

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
        sc.stop();
    }
}

/*
package SparkDeepLearning.SparkDeepLearning;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    private int batchSize = 128;
    private int numEpochs = 2;

    public static void main(String[] args) throws Exception {
        new App().entryPoint();
    }

    protected void entryPoint() throws Exception {
        // Load the data using DataSetIterator
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSize, false, 12345);

        // Create network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(30).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(30).nOut(10).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                .build();

        // Create and train the network
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < numEpochs; i++) {
            net.fit(iterTrain);
            log.info("Completed Epoch {}", i);
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime; // Time taken in milliseconds

        log.info("Training Time: {} milliseconds", duration);

        // Save the model
        net.save(new File("/home/cluster/model4.bin"));

        // Load the model for evaluation
        MultiLayerNetwork netForEval = MultiLayerNetwork.load(new File("/home/cluster/model4.bin"), true);

        // Perform evaluation
        Evaluation evaluation = netForEval.evaluate(iterTest);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        log.info("***** Example Complete *****");
    }
}




*/
/*
package SparkDeepLearning.SparkDeepLearning;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class App {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    private int batchSize = 128;
    private int numEpochs = 2;

    public static void main(String[] args) throws Exception {
        new App().entryPoint();
    }

    protected void entryPoint() throws Exception {
        // Load the data using DataSetIterator
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSize, false, 12345);

        // Create network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(30).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(30).nOut(10).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(10).build())
                .build();

        // Create and train the network
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < numEpochs; i++) {
            net.fit(iterTrain);
            log.info("Completed Epoch {}", i);
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime; // Time taken in milliseconds

        log.info("Training Time: {} milliseconds", duration);

        // Save the model
        net.save(new File("model4.bin"));

        // Load the model for evaluation
        MultiLayerNetwork netForEval = MultiLayerNetwork.load(new File("model4.bin"), true);

        // Perform evaluation
        Evaluation evaluation = netForEval.evaluate(iterTest);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        log.info("***** Example Complete *****");
    }
}

*/