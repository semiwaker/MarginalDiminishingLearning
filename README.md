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
| Baseline unbiased | 0.9707 | 0.5569 | 0.65, 0.61, 0.40, 0.44, 0.45, 0.40, 0.91, 0.52, 0.64, 0.47 |
| Baseline biased | 0.9764 | 0.4792 | 0.78, 0.84, 0.56, 0.48, 0.60, 0.32, 0.61, 0.35, 0.22, 0.001|


