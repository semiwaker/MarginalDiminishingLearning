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

Run 10 trials to get average result.

| Situation | train | val | cates |
| ---- | ---- | ---- | --- |
| Baseline unbiased | 0.9344 | 0.5696 | 0.61, 0.67, 0.52, 0.38, 0.53, 0.37, 0.77, 0.54, 0.68, 0.51 |
| Baseline biased | 0.9377 | 0.4311 | 0.75, 0.75, 0.65, 0.46, 0.38, 0.25, 0.56, 0.25, 0.19, 0.003 |
| Power law weighted, $\alpha=-0.5$ | 0.9332 | 0.4388 | 0.73, 0.73, 0.58, 0.49, 0.51, 0.26, 0.56, 0.29, 0.17, 0.003 |
| Power law weighted, $\alpha=-1$ | 0.9380 | 0.4413 | 0.72, 0.72, 0.55, 0.52, 0.47, 0.25, 0.62, 0.31, 0.17, 0.004 |
| Power law weighted, $\alpha=-2$ | 0.9477 | 0.4529 | 0.71, 0.77, 0.61, 0.47, 0.44, 0.31, 0.58, 0.33, 0.23, 0.005 |
| Power law weighted, $\alpha=-5$ | 0.1553 | 0.1136 | 0.13, 0.13, 0.12, 0.13, 0.10, 0.10, 0.11, 0.08, 0.07, 0.075 |
| Kmeans weigthed, $\alpha=-0.5$ | 0.8934 | 0.4330 | 0.72, 0.70, 0.60, 0.44, 0.50, 0.24, 0.56, 0.29, 0.19, 0.004 |
| Kmeans weigthed, $\alpha=-0.5$, warm up | 0.8595 | 0.4176 | 0.72, 0.63, 0.57, 0.44, 0.48, 0.28, 0.54, 0.27, 0.15, 0.002 |
| Kmeans weigthed, $\alpha=-1$, | 0.9028 | 0.4474 | 0.71, 0.71, 0.59, 0.46, 0.52, 0.32, 0.57, 0.30, 0.18, 0.002 |
| Kmeans weigthed, $\alpha=-1$, warm up | 0.8755 | 0.4262 | 0.73, 0.65, 0.53, 0.52, 0.47, 0.27, 0.55, 0.27, 0.16, 0.003 |
| Kmeans weigthed, $\alpha=-2$ | 0.7655 | 0.4030 | 0.65, 0.64, 0.59, 0.48, 0.41, 0.27, 0.37, 0.24, 0.16, 0.007 |
| Kmeans weigthed, $\alpha=-2$, warm up | 0.7924 | 0.3998 | 0.70, 0.64, 0.58, 0.42, 0.40, 0.26, 0.42, 0.21, 0.22, 0.004 |
| Kmeans with distance, $\alpha=-1$ | 0.8690 | 0.4217 | 0.67, 0.69, 0.60, 0.46, 0.47, 0.22, 0.53, 0.29, 0.18, 0.003 |
