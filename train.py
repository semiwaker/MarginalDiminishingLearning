import os.path as path
import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import scipy.cluster as cluster

import option
import data
import model
import utils


def powLawOnLabels(labels, numLabels, params):
    cnt = [0] * numLabels
    for i in labels:
        cnt[i] += 1
    label_weight = np.array(cnt)
    label_weight = np.power(label_weight, params.alpha)
    label_weight /= np.sum(label_weight)
    weights = [label_weight[i] for i in labels]
    return weights


def trainBaseline(params):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    baseline = model.makeBaselineModel(params)
    baseline.summary()

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()
    baseline.compile(optimizer, loss=loss_fn, metrics=[metricAcc, cateAcc])

    baseline.fit(trainSet, epochs=params.numEpochs, validation_data=testSet)

    baseline.save_weights(path.join(params.modelPath, "baseline.keras"))

    accs = baseline.evaluate(valSet)
    print(accs)


def trainWeightedBaseline(params):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    labels = dataloader.trainData[b"labels"]
    weights = powLawOnLabels(labels, 10, params)

    baseline = model.makeBaselineModel(params)
    baseline.summary()
    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()
    baseline.compile(optimizer, loss=loss_fn, metrics=[
                     metricAcc, cateAcc], loss_weights=weights)

    baseline.fit(trainSet, epochs=params.numEpochs, validation_data=testSet)

    baseline.save_weights(
        path.join(params.modelPath, "weightedBaseline.keras"))

    accs = baseline.evaluate(valSet)
    for i in accs[2]:
        print("%.2lf," % i, end=' ')
    print()
    print("%.3lf" % accs[2][9])


def KMeanupdate(encode_record, params):
    centroids, label = cluster.vq.kmeans2(encode_record, 10, minit="points")
    weights = powLawOnLabels(label, 10, params)
    return tf.constant(weights, dtype=tf.float32)


def trainNeighbourWeight(params, weightUpdateFunc):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    splitModel = model.makeSplitModel(params)

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()

    datalen = dataloader.trainLen
    weights = tf.ones([datalen], dtype=tf.float32)
    timer = utils.Timer()
    # trainables = [splitModel.conv.trainable_variables,
    #   splitModel.predict.trainable_variables]
    for epochID in range(1, params.numEpochs+1):
        IDcnt = 0
        batchCnt = 0
        metricAcc.reset_states()
        trainloss = 0
        encode_record = []
        for batch, labels in trainSet:
            batchSize = batch.shape[0]
            batchWeight = weights[IDcnt: IDcnt+batchSize]
            with tf.GradientTape() as tape:
                prediction, encoding = splitModel(batch, training=True)
                loss = loss_fn(labels, prediction, batchWeight)
            trainloss += loss
            gradient = tape.gradient(loss, splitModel.trainable_variables)
            optimizer.apply_gradients(
                zip(gradient, splitModel.trainable_variables))
            metricAcc.update_state(labels, prediction)

            encode_record.append(encoding.numpy())

            batchCnt += 1
            IDcnt += batchSize
        trainAcc = metricAcc.result()
        trainloss /= batchCnt

        batchCnt = 0
        testloss = 0
        metricAcc.reset_states()
        for batch, labels in testSet:
            prediction, _ = splitModel(batch, training=False)
            loss = loss_fn(labels, prediction)
            testloss += loss
            batchCnt += 1
            metricAcc.update_state(labels, prediction)
        testAcc = metricAcc.result()
        testloss /= batchCnt

        print(f"Epoch {epochID} {timer()}")
        print(f"Train Loss: {trainloss} Accuracy: {trainAcc}")
        print(f"Test Loss: {testloss} Accuracy: {testAcc}")

        if epochID >= params.warmUpEpochs:
            encode_record = np.concatenate(encode_record)
            weights = weightUpdateFunc(encode_record, params)

    splitModel.save_weights(
        path.join(params.modelPath, "splitModel.keras"))
    metricAcc.reset_states()
    cateAcc.reset_states()
    for batch, labels in valSet:
        prediction, _ = splitModel(batch, training=False)
        metricAcc.update_state(labels, prediction)
        cateAcc.update_state(labels, prediction)
    acc = cateAcc.result()
    print(metricAcc.result())
    for i in acc:
        print("%.2lf," % i, end=' ')
    print()
    print("%.3lf" % acc[9])


if __name__ == "__main__":
    params = option.read()

    trainFuncs = {
        "None": lambda x: None,
        "baseline": trainBaseline,
        "weightedBaseline": trainWeightedBaseline,
        "kmean": lambda x: trainNeighbourWeight(x, KMeanupdate)
    }
    if params.useGPU:
        utils.selectDevice(0)
    trainFuncs[params.trainModel](params)
