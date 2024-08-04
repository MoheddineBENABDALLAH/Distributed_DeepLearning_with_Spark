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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class test {
    private static final Logger log = LoggerFactory.getLogger(App.class);

    private int batchSize = 64;
    private int numEpochs = 100;

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
            log.info("Completed Epoch {}", i);
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime; // Time taken in milliseconds

        // Save the model

        // Load the model for evaluation
        MultiLayerNetwork restoredModel = MultiLayerNetwork.load(new File("/home/user/model2.bin"), true);

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
