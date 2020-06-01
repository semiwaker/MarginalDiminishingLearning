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

Number of instances of each category in the biased train data: 1000, 950, 950, 900, 500, 450, 400, 200, 100, 10

| Situation | train | val | cates |
| ---- | ---- | ---- | --- |
| Baseline unbiased | 0.9707 | 0.5569 | 0.65, 0.61, 0.40, 0.44, 0.45, 0.40, 0.91, 0.52, 0.64, 0.47 |
| Baseline biased | 0.9764 | 0.4792 | 0.78, 0.84, 0.56, 0.48, 0.60, 0.32, 0.61, 0.35, 0.22, 0.001 |
| Power law weighted, $\alpha=-0.5$ | 0.9507 | 0.4410 | 0.82, 0.75, 0.58, 0.46, 0.50, 0.24, 0.61, 0.28, 0.10, 0.01 |
| Power law weighted, $\alpha=-1$ | 0.9628 | 0.4544 | 0.74, 0.67, 0.69, 0.44, 0.38, 0.32, 0.56 , 0.37, 0.30 , 0.002 |
| Power law weighted, $\alpha=-2$ | 0.9641 | 0.4325 | 0.74, 0.83, 0.63, 0.32, 0.33, 0.12, 0.81, 0.32, 0.20, 0.002 |
| Power law weighted, $\alpha=-5$(slow to converge) | 0.1984 | 0.1336 | 0.23, 0.20, 0.15, 0.13, 0.11, 0.09, 0.07, 0.04, 0.09, 0.04 |
| Kmeans weigthed, $\alpha=-0.5$ | 0.9476 | 0.3888 | 0.70, 0.58, 0.79, 0.30, 0.29, 0.10, 0.56, 0.20, 0.16, 0.002 |
| Kmeans weigthed, $\alpha=-0.5$, warm up | 0.9271 | 0.4047 | 0.88, 0.60, 0.59, 0.34, 0.48, 0.17, 0.37, 0.12, 0.34, 0.004 |
| Kmeans weigthed, $\alpha=-1$, warm up | 0.9692 | 0.4433 | 0.76, 0.79, 0.58, 0.52, 0.43, 0.37, 0.46, 0.39, 0.09, 0.01 |
| Power law loss weighted, $\alpha=1$, warm up | 0.8540 | 0.4359 | 0.77, 0.79, 0.59, 0.50, 0.37, 0.19, 0.42, 0.34, 0.18, 0.005 |
| Power law loss weighted, $\alpha=2$, warm up | 0.7410 | 0.3992 | 0.76, 0.63, 0.58, 0.33, 0.38, 0.19, 0.58, 0.14, 0.12, 0.013 |
| Power law loss weighted, $\alpha=0.1$, warm up | 0.9557 | 0.4510 | 0.74, 0.72, 0.57, 0.53, 0.46, 0.31, 0.51, 0.35, 0.21, 0.005 |
| Power law loss weighted, $\alpha=0.2$, warm up | 0.9380 | 0.4452 | 0.73, 0.74, 0.59, 0.44, 0.45, 0.35, 0.57, 0.29, 0.19, 0.005 |
| Power law loss weighted, $\alpha=0.3$, warm up | 0.9022 | 0.4223 | 0.69, 0.70, 0.53, 0.55, 0.48, 0.27, 0.47, 0.32, 0.12, 0.007 |
| Power law trend loss weighted, $\alpha=1$, warm up | 0.7730 | 0.3871 | 0.66, 0.58, 0.56, 0.42, 0.46, 0.18, 0.52, 0.23, 0.11, 0.009 |
| Power law trend loss weighted, $\alpha=0.5$, warm up | 0.8999 | 0.4209 | 0.71, 0.70, 0.60, 0.44, 0.45, 0.28, 0.46, 0.25, 0.17, 0.008 |
| Power law trend loss weighted, $\alpha=0.3$, warm up | 0.9535 | 0.4433 | 0.77, 0.76, 0.56, 0.50, 0.48, 0.29, 0.50, 0.30, 0.14, 0.006 |
| Power law trend loss weighted, $\alpha=0.2$, warm up | 0.9330 | 0.4380 | 0.75, 0.72, 0.57, 0.47, 0.49, 0.26, 0.60, 0.31, 0.13, 0.004 |
| Power law trend loss weighted, $\alpha=0.1$, warm up | 0.9447 | 0.4432 | 0.77, 0.73, 0.59, 0.45, 0.49, 0.27, 0.54, 0.35, 0.15, 0.002 |
| Power law featureL1 weighted, $\alpha=1$, warm up | 0.9574 | 0.4515 | 0.75, 0.75, 0.60, 0.48, 0.45, 0.25, 0.63, 0.33, 0.21, 0.005 |
| Power law featureL1 weighted, $\alpha=2$, warm up | 0.9111 | 0.4296 | 0.74, 0.73, 0.55, 0.47, 0.50, 0.24, 0.59, 0.27, 0.14, 0.001 |
| Power law featureL1 weighted, $\alpha=0.2$, warm up | 0.9336 | 0.4407 | 0.75, 0.75, 0.60, 0.45, 0.48, 0.22, 0.60, 0.29, 0.18, 0.003 |

