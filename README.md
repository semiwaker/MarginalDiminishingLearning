# MarginalDiminishingLearning

This is a project to study the marginal diminishing effect on the utility of learning.

## Dataset

[Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)

[Python pickle version](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

Use the Python version of CIFAR10.
Run Dataloader.makeBalanced first to create balanced datasets `data_bal_1` and `data_bal_2`.
Use `data_bal_1` as training set, `data_bal_2` as test set and `test_batch` as val set.
Run Dataloader.makeSplit to create split file to specify number of samples in each catergory.

## Power law of learning

$\ln y = a\ln x + b$, can be simplified to $y = \beta\cdot x^\alpha,\beta > 0$

## result

Number of instances of each category in the biased train data:
| Label | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Instances | 1000 | 950 | 950 | 900 | 500 | 450 | 400 | 200 | 100 | 10 |

Run 5 trials to get average result.

| Situation | train | val | cate recall |
| ---- | ---- | ---- | --- |
| Baseline unbiased | 0.9344 | 0.5696 | 0.61, 0.67, 0.52, 0.38, 0.53, 0.37, 0.77, 0.54, 0.68, 0.51 |
| Baseline biased | 0.9377 | 0.4311 | 0.75, 0.75, 0.65, 0.46, 0.38, 0.25, 0.56, 0.25, 0.19, 0.003 |
| Power law weighted, $\alpha=-0.5,\beta=10$ | 0.9263 | 0.4332 | 0.78, 0.70, 0.63, 0.49, 0.42, 0.23, 0.54, 0.33, 0.12, 0.011 |
| Power law weighted, $\alpha=-1,\beta=40$ | 0.8733 | 0.4161 | 0.73, 0.70, 0.59, 0.47, 0.45, 0.21, 0.50, 0.26, 0.19, 0.001 |
| Power law weighted, $\alpha=-2,\beta=300$ | 0.8785 | 0.4137 | 0.75, 0.68, 0.58, 0.46, 0.48, 0.30, 0.48, 0.21, 0.15, 0.002 |
| Power law weighted, $\alpha=-5,\beta=400$ | 0.8081 | 0.3937 | 0.69, 0.68, 0.61, 0.39, 0.37, 0.20, 0.57, 0.25, 0.14, 0.002 |
| Kmeans weigthed, $\alpha=-0.5,\beta=10$ | 0.7463 | 0.4044 | 0.64, 0.58, 0.57, 0.42, 0.50, 0.24, 0.55, 0.26, 0.11, 0.003 |
| Kmeans weigthed, $\alpha=-0.5,\beta=10$, warm up | 0.7984 | 0.4177 | 0.76, 0.62, 0.52, 0.50, 0.56, 0.26, 0.40, 0.28, 0.14, 0.001 |
| Kmeans weigthed, $\alpha=-1,\beta=10$, | 0.8559 | 0.4123 | 0.73, 0.71, 0.58, 0.46, 0.45, 0.24, 0.46, 0.22, 0.20, 0.004 |
| Kmeans weigthed, $\alpha=-1,\beta=10$, warm up | 0.7555 | 0.4035 | 0.72, 0.62, 0.59, 0.43, 0.42, 0.24, 0.54, 0.21, 0.10, 0.004 |
| Kmeans weigthed, $\alpha=-2,\beta=10$, | 0.6710 | 0.3804 | 0.63, 0.60, 0.55, 0.45, 0.44, 0.22, 0.38, 0.15, 0.11, 0.003 |
| Kmeans weigthed, $\alpha=-2,\beta=10$, warm up | 0.7102 | 0.4073 | 0.74, 0.61, 0.55, 0.39, 0.47, 0.18, 0.60, 0.21, 0.14, 0.002 |
| Kmeans weigthed, $\alpha=-5,\beta=10$ | 0.4561 | 0.2663 | 0.61, 0.47, 0.52, 0.26, 0.21, 0.15, 0.13, 0.04, 0.05, 0.001 |
| Kmeans weigthed, $\alpha=-5,\beta=10$, warm up | 0.6135 | 0.3674 | 0.67, 0.66, 0.55, 0.35, 0.35, 0.17, 0.34, 0.15, 0.09, 0.002 |
| Kmeans with distance, $\alpha=-0.5,\beta=10$ | 0.7754 | 0.4098 | 0.70, 0.63, 0.56, 0.45, 0.44, 0.25, 0.59, 0.26, 0.11, 0.002 |
| Kmeans with distance, $\alpha=-0.5,\beta=10$, warm up | 0.7916 | 0.4150 | 0.70, 0.66, 0.64, 0.40, 0.47, 0.18, 0.55, 0.23, 0.20, 0.002 |
| Kmeans with distance, $\alpha=-1,\beta=2$ | 0.7284 | 0.4094 | 0.75, 0.64, 0.52, 0.38, 0.46, 0.26, 0.50, 0.21, 0.15, 0.005 |
| Kmeans with distance, $\alpha=-1,\beta=2$, warm up | 0.7361 | 0.3974 | 0.69, 0.63, 0.56, 0.47, 0.44, 0.22, 0.44, 0.21, 0.15, 0.003 |
| Kmeans with distance, $\alpha=-2,\beta=50$ | 0.5916 | 0.3425 | 0.70, 0.58, 0.50, 0.34, 0.29, 0.24, 0.23, 0.06, 0.10, 0.005 |
| Kmeans with distance, $\alpha=-2,\beta=50$, warm up | 0.6735 | 0.3902 | 0.70, 0.59, 0.55, 0.39, 0.43, 0.21, 0.37, 0.17, 0.14, 0.004 |
| Kmean batch update, $\alpha=-1,\beta=10$| 0.6906 | 0.3602 | 0.68, 0.51, 0.60, 0.45, 0.41, 0.15, 0.52, 0.13, 0.10, 0.002 |
| Kmean batch update, $\alpha=-1,\beta=10$, warm up| 0.7561 | 0.3993 | 0.74, 0.66, 0.61, 0.36, 0.40, 0.24, 0.49, 0.18, 0.15, 0.003 |
| Kmean batch update, $\alpha=-2,\beta=10$| 0.7758 | 0.4299 | 0.72, 0.62, 0.54, 0.50, 0.44, 0.34, 0.50, 0.28, 0.13, 0.003 |
| Kmean batch update, $\alpha=-2,\beta=10$, warm up| 0.7210 | 0.4056 | 0.68, 0.60, 0.55, 0.40, 0.59, 0.27, 0.38, 0.28, 0.07, 0.002 |
| Power law loss weighted, $\alpha=1$, warm up (bug)| 0.8540 | 0.4359 | 0.77, 0.79, 0.59, 0.50, 0.37, 0.19, 0.42, 0.34, 0.18, 0.005 |
| Power law loss weighted, $\alpha=1$, warm up | 0.8477 | 0.4101 | 0.63, 0.71, 0.56, 0.48, 0.48, 0.22, 0.60, 0.22, 0.15, 0.001 |
| Power law loss weighted, $\alpha=2$, warm up (bug)| 0.7410 | 0.3992 | 0.76, 0.63, 0.58, 0.33, 0.38, 0.19, 0.58, 0.14, 0.12, 0.013 |
| Power law loss weighted, $\alpha=0.1$, warm up (bug)| 0.9557 | 0.4510 | 0.74, 0.72, 0.57, 0.53, 0.46, 0.31, 0.51, 0.35, 0.21, 0.005 |
| Power law loss weighted, $\alpha=0.1$, warm up | 0.8913 | 0.4181 | 0.70, 0.67, 0.59, 0.44, 0.46, 0.30, 0.57, 0.26, 0.12, 0.001 |
| Power law loss weighted, $\alpha=0.2$, warm up | 0.8723 | 0.4192 | 0.72, 0.62, 0.59, 0.48, 0.47, 0.30, 0.56, 0.27, 0.13, 0.002 |
| Power law loss weighted, $\alpha=0.3$, warm up (bug)| 0.9022 | 0.4223 | 0.69, 0.70, 0.53, 0.55, 0.48, 0.27, 0.47, 0.32, 0.12, 0.007 |
| Power law loss weighted, $\alpha=0.5$, warm up | 0.8846 | 0.4139 | 0.71, 0.74, 0.63, 0.41, 0.38, 0.21, 0.51, 0.22, 0.24, 0.002 |
| Power law trend loss weighted, $\alpha=1$, warm up (bug)| 0.7730 | 0.3871 | 0.66, 0.58, 0.56, 0.42, 0.46, 0.18, 0.52, 0.23, 0.11, 0.009 |
| Power law trend loss weighted, $\alpha=1$, warm up | 0.8415 | 0.3947 | 0.73, 0.61, 0.60, 0.39, 0.35, 0.28, 0.44, 0.23, 0.15, 0.008 |
| Power law trend loss weighted, $\alpha=0.5$, warm up (bug)| 0.8999 | 0.4209 | 0.71, 0.70, 0.60, 0.44, 0.45, 0.28, 0.46, 0.25, 0.17, 0.008 |
| Power law trend loss weighted, $\alpha=0.5$, warm up | 0.8349 | 0.3984 | 0.70, 0.60, 0.60, 0.43, 0.42, 0.22, 0.50, 0.25, 0.15, 0.002 |
| Power law trend loss weighted, $\alpha=0.3$, warm up (bug)| 0.9535 | 0.4433 | 0.77, 0.76, 0.56, 0.50, 0.48, 0.29, 0.50, 0.30, 0.14, 0.006 |
| Power law trend loss weighted, $\alpha=0.2$, warm up (bug)| 0.9330 | 0.4380 | 0.75, 0.72, 0.57, 0.47, 0.49, 0.26, 0.60, 0.31, 0.13, 0.004 |
| Power law trend loss weighted, $\alpha=0.2$, warm up | 0.8632 | 0.3981 | 0.65, 0.62, 0.60, 0.50, 0.44, 0.29, 0.50, 0.23, 0.10, 0.002 |
| Power law trend loss weighted, $\alpha=0.1$, warm up (bug)| 0.9447 | 0.4432 | 0.77, 0.73, 0.59, 0.45, 0.49, 0.27, 0.54, 0.35, 0.15, 0.002 |
| Power law trend loss weighted, $\alpha=0.1$, warm up | 0.8867 | 0.4250 | 0.65, 0.74, 0.59, 0.42, 0.50, 0.26, 0.62, 0.27, 0.13, 0.002 |
| Power law featureL1 weighted, $\alpha=1$, warm up (bug)| 0.9574 | 0.4515 | 0.75, 0.75, 0.60, 0.48, 0.45, 0.25, 0.63, 0.33, 0.21, 0.005 |
| Power law featureL1 weighted, $\alpha=1$, warm up | 0.9153 | 0.4311 | 0.71, 0.70, 0.63, 0.47, 0.43, 0.27, 0.56, 0.23, 0.26, 0.003 |
| Power law featureL1 weighted, $\alpha=2$, warm up (bug)| 0.9111 | 0.4296 | 0.74, 0.73, 0.55, 0.47, 0.50, 0.24, 0.59, 0.27, 0.14, 0.001 |
| Power law featureL1 weighted, $\alpha=0.5$, warm up | 0.9080 | 0.4306 | 0.64, 0.74, 0.62, 0.47, 0.52, 0.24, 0.55, 0.29, 0.18, 0.003 |
| Power law featureL1 weighted, $\alpha=0.2$, warm up (bug)| 0.9336 | 0.4407 | 0.75, 0.75, 0.60, 0.45, 0.48, 0.22, 0.60, 0.29, 0.18, 0.003 |
| Power law featureL1 weighted, $\alpha=0.2$, warm up | 0.8682 | 0.4142 | 0.69, 0.69, 0.67, 0.40, 0.47, 0.20, 0.59, 0.25, 0.13, 0.004 |
| Power law featureL1 weighted, $\alpha=0.1$, warm up | 0.8548 | 0.4083 | 0.67, 0.70, 0.58, 0.41, 0.48, 0.33, 0.48, 0.28, 0.09, 0.001 |
