![alt text](https://miro.medium.com/v2/resize:fit:640/format:webp/1*691Sexy23zPn0Mv_T6pgBQ.png)

Deeplearning4j is 

Works as job within Spark :

1- Shards of the input dataset are distrubted over all cores
2- Workers process data synchronously in parralel
3- A model is trained on each shard of the input dataset
4- Workers send the transformed parameters of their models back to the master
5- the master averages the parametres
6- the parametres are used to update the model on each worker's core
7- When the error ceases to shrink,the Spark job ends
