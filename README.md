# Distributed Deep Learning with Spark and Deeplearning4j

This project demonstrates how to perform distributed deep learning using Spark and Deeplearning4j. The aim is to leverage the power of parallel processing to train models on large datasets efficiently.

## Introduction

Distributed deep learning allows the training of machine learning models on large datasets by dividing the workload across multiple cores and machines. This project uses Spark to distribute the data and Deeplearning4j to train the models. The process involves the following steps:

1. **Shards of the input dataset are distributed over all cores.**
2. **Workers process data synchronously in parallel.**
3. **A model is trained on each shard of the input dataset.**
4. **Workers send the transformed parameters of their models back to the master.**
5. **The master averages the parameters.**
6. **The parameters are used to update the model on each worker's core.**
7. **When the error ceases to shrink, the Spark job ends.**

## Architecture of Data Parallelism in Distributed Training

![Architecture of Data Parallelism in Distributed Training](https://miro.medium.com/v2/resize:fit:640/format:webp/1*691Sexy23zPn0Mv_T6pgBQ.png)

## Architecture of Distributed Training with Spark

![Architecture of Distributed Training with Spark](https://static001.infoq.cn/resource/image/f2/74/f2148eb24747d930ef6faffe5fa90674.png)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Java 8 or higher**
- **Apache Spark 3.0 or higher**
- **Maven**
- **Deeplearning4j**

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/MoheddineBENABDALLAH/Spark_DeepLearning.git
    cd Spark_DeepLearning
    ```

2. **Build the project using Maven:**

    ```bash
    mvn clean install
    ```

3. **Ensure you have Apache Spark set up and configured correctly.**

## Usage

1. **Prepare your data and place it in a suitable directory.**

2. **Configure your Spark and Deeplearning4j settings as needed.**

3. **Run the distributed training job:**

    ```bash
    spark-submit --class SparkDeepLearning.SparkDeepLearning.App --master <spark-master-url> target/SparkDeepLearning-0.0.1-SNAPSHOT.jar
    ```


