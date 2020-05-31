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
| Power law weighted, $\alpha=-1$ | 0.9628 | 0.4544 | 0.74, 0.67, 0.69, 0.44, 0.38, 0.32, 0.56 , 0.37, 0.30 , 0.002 |
| Power law weighted, $\alpha=-2$ | 0.9641 | 0.4325 | 0.74, 0.83, 0.63, 0.32, 0.33, 0.12, 0.81, 0.32, 0.20, 0.002 |
| Power law weighted, $\alpha=-5$(slow to converge) | 0.1984 | 0.1336 | 0.23, 0.20, 0.15, 0.13, 0.11, 0.09, 0.07, 0.04, 0.09, 0.04 |
| Kmeans weigthed, $\alpha=-0.5$ | 0.9476 | 0.3888 | 0.70, 0.58, 0.79, 0.30, 0.29, 0.10, 0.56, 0.20, 0.16, 0.002 |
| Power law loss weighted, $\alpha=1$ | 0.8540 | 0.4359 | 0.77, 0.79, 0.59, 0.50, 0.37, 0.19, 0.42, 0.34, 0.18, 0.00, 0.005 |
| Power law loss weighted, $\alpha=2$ | 0.7410 | 0.3992 | 0.76, 0.63, 0.58, 0.33, 0.38, 0.19, 0.58, 0.14, 0.12, 0.01, 0.013 |
| Power law loss weighted, $\alpha=0.5$ | 0.9042 | 0.4139 | 0.68, 0.65, 0.71, 0.49, 0.32, 0.20, 0.54, 0.32, 0.12, 0.01, 0.005 |
| Power law loss weighted, $\alpha=0.1$ | 0.9548 | 0.4623 | 0.68, 0.80, 0.59, 0.55, 0.56, 0.23, 0.64, 0.31, 0.15, 0.00, 0.003 |
