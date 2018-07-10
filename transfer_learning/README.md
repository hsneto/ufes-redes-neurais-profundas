# Transfer Learning

*Forked from [Transfer-Learning](https://github.com/clebeson/Deep_Learning/tree/master/Transfer-Learning).

## Database
![CIFAR10](https://github.com/hsneto/redes_neurais_profundas/blob/master/.docs/images/cifar10.png)
  
Figure 1: A sample of CIFAR 10 dataset

## Model
![VGG16](https://github.com/hsneto/redes_neurais_profundas/blob/master/.docs/images/vgg16.png)
 
Figure 2: The VGG16 architecture

---
### Exercise 1
Comparisons between the following parameters:
 - bottleneck
 - fine tuning
 - initial learning rate
 - data augmentation
 - model layers

#### Data summary

```
data1 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-1
```
```
data2 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-3
```
```
data3 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-5
```
```
data4 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-3 | keep = 0.5
```
```
data5 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-3 | data_augmentation = False
```
```
data6 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-3 | bottleneck = False
```
```
data7 ---> cutting_layer = 'pool4' | initial_learning_rate = 1e-3 | fine_tunning = True
```
```
data8 ---> cutting_layer = 'pool3' | initial_learning_rate = 1e-3
```
```
data9 ---> cutting_layer = 'pool5' | initial_learning_rate = 1e-3
```

![Accuracy Analysis](https://github.com/hsneto/redes_neurais_profundas/blob/master/transfer_learning/results/exe_1/accuracy_analysis.png)
 
Figure 3: Parameters Analysis based on accuracy

![Elapsed time Analysis](https://github.com/hsneto/redes_neurais_profundas/blob/master/transfer_learning/results/exe_1/elapsed_time_Analysis.png)
 
Figure 4: Parameters Analysis based on training time

![Confusion Matrix](https://github.com/hsneto/redes_neurais_profundas/blob/master/transfer_learning/results/exe_1/cm_data8.png)
 
Figure 5: Confusion Matrix from the model with the best accuracy (data8)

### Exercise 2
 - Generate validation dataset
 - Implementation of "early stop"

![Losses -wo early stop](https://github.com/hsneto/redes_neurais_profundas/blob/master/transfer_learning/results/exe_2/losses_normal.png)
 
Figure 6: Transfer Learning training without `early stop`

![Losses -w early stop](https://github.com/hsneto/redes_neurais_profundas/blob/master/transfer_learning/results/exe_2/losses_early_stop.png)
 
Figure 7: Transfer Learning training with `early stop`

### Exercise 3
Calculate the following metrics:
 - Accuracy
 - Recall
 - Precision
 - F1-score

--- 
*all transfer learning exercises were developed using the [docker container](https://github.com/hsneto/redes_neurais_profundas#docker)
