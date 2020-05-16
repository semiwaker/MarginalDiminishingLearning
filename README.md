# MarginalDiminishingLearning

This is a project to study the marginal diminishing effect on the utility of learning.

## Dataset

[Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)

[Python pickle version](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

Use the Python version of CIFAR10.
Run Dataloader.makeBalanced first to create balanced datasets `data_bal_1` and `data_bal_2`.
Use `data_bal_1` as training set, `data_bal_2` as test set and `test_batch` as val set.
Run Dataloader.makeSplit to create split file to specify number of samples in each catergory.

## result

| Situation | train | val | cates |
| ---- | ---- | ---- | --- |
| Baseline unbiased | 0.9112 | 0.5497 | 0.54, 0.66, 0.48, 0.24, 0.44, 0.27, 0.79, 0.59, 0.71, 0.57 |
| Baseline biased | 0.9500 | 0.4475 | 0.79, 0.83, 0.51, 0.50, 0.34, 0.38, 0.49, 0.37, 0.18, 0.005|
