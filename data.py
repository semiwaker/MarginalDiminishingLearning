import pickle
import json
import itertools
import os.path as path
import random

import tensorflow as tf
import tensorflow.keras as keras

import option


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.dataset = params.dataset

    def readData(self):
        self.meta = self.loadFile(path.join(self.dataset, "batches.meta"))
        self.labelNames = [str(s, encoding="utf-8")
                           for s in self.meta[b"label_names"]]

        self.trainData = self.loadFile(path.join(self.dataset, "data_bal_1"))
        self.testData = self.loadFile(path.join(self.dataset, "data_bal_2"))
        self.valData = self.loadFile(path.join(self.dataset, "test_batch"))
        with open(self.params.splitPath, "r") as file:
            self.split = json.load(file)

        def applySplit(dataset, bias):
            data = []
            label = []
            for i in range(10):
                d = dataset[b"data"][i * 1000: i * 1000 + 1000]
                l = dataset[b"labels"][i * 1000: i * 1000 + 1000]
                if bias:
                    data.extend(d[:self.split[i]])
                    label.extend(l[:self.split[i]])
                else:
                    data.extend(d)
                    label.extend(l)
            tmp = list(zip(data, label))
            random.shuffle(tmp)
            data = [i[0] for i in tmp]
            label = [i[1] for i in tmp]
            return {b"data": data, b"labels": label}

        if not self.params.dataUnbias:
            self.trainData = applySplit(self.trainData, True)
        else:
            self.trainData = applySplit(self.trainData, False)

        # Keep testData and valData unchanged
        self.testData = applySplit(self.testData, False)
        self.valData = applySplit(self.valData, False)

        self.trainLen = len(self.trainData[b"data"])
        self.testLen = len(self.testData[b"data"])
        self.valLen = len(self.valData[b"data"])

        def makeDataset(data):
            dataset = tf.data.Dataset.from_tensor_slices(data[b"data"])
            dataset = dataset.map(lambda x: tf.reshape(x, [3, 32, 32]))
            dataset = dataset.map(lambda x: tf.stack(
                [x[0], x[1], x[2]], axis=-1))
            dataset = dataset.map(lambda x: tf.cast(x/256, tf.float32))
            labelset = tf.data.Dataset.from_tensor_slices(data[b"labels"])
            labelset = labelset.map(lambda x: tf.one_hot(x, 10))
            dataset = tf.data.Dataset.zip((dataset, labelset))
            dataset = dataset.batch(self.params.batchSize)
            dataset = dataset.prefetch(2)
            return dataset
        self.trainSet, self.testSet, self.valSet = map(
            makeDataset, (self.trainData, self.testData, self.valData))

        return self.trainSet, self.testSet, self.valSet

    def loadFile(self, file):
        with open(file, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d

    def makeSplit(self):
        split = [1000, 950, 950, 900, 500, 450, 400, 200, 100, 10]
        with open(self.params.splitPath, "w") as file:
            json.dump(split, file, indent=4)

    def makeBalanced(self):
        data = [[] for i in range(10)]

        for i in range(1, 6):
            d = dataloader.loadFile(
                "data/cifar-10-batches-py/data_batch_"+str(i))
            label = d[b"labels"]
            for i in range(10000):
                data[label[i]].append(d[b"data"][i])

        data1 = {
            b"data": list(itertools.chain(*[u[:1000] for u in data])),
            b"labels": [i for i in range(10) for j in range(1000)]
        }
        data2 = {
            b"data": list(itertools.chain(*[u[1000:2000] for u in data])),
            b"labels": [i for i in range(10) for j in range(1000)]
        }

        with open("data/cifar-10-batches-py/data_bal_1", "wb") as file:
            pickle.dump(data1, file)
        with open("data/cifar-10-batches-py/data_bal_2", "wb+") as file:
            pickle.dump(data2, file)


if __name__ == "__main__":
    params = option.read()
    dataloader = DataLoader(params)
    train, test, val = dataloader.readData()
    for data, label in train.take(1):
        print(data)
        print(label)
