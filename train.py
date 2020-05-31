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

def avgAcc(numTrial, dataloader, trainFunc, params):
    trainSet, testSet, valSet = dataloader.readData()
    trainAcc = []
    valAcc = []
    cateAcc = []
    timer = utils.Timer()
    for i in range(1, numTrial+1):
        print(f"Trial {i} starts at {timer()}")
        ta, va, ca = trainFunc(dataloader, trainSet, testSet, valSet, params)
        print(f"Trail {i} result:")
        print(f"Train accuracy {ta}")
        print(f"Val accuracy {va}")
        print(f"Cate accuracy {ca}")
        trainAcc.append(ta)
        valAcc.append(va)
        cateAcc.append(ca)
    trainAcc = tf.reduce_mean(tf.stack(trainAcc))
    valAcc = tf.reduce_mean(tf.stack(valAcc))
    cateAcc = tf.reduce_mean(tf.stack(cateAcc), axis=0)
    print(f"Finished {timer()}")
    print(f"Overall train accuracy {trainAcc}")
    print(f"Overall val accuracy {valAcc}")
    print(f"Overall cate accuracy ", end='')
    for i in cateAcc[:-1]:
        print("%.2lf," % i, end=' ')
    print("%.3f" % cateAcc[-1])


def trainBaseline(dataloader, trainSet, testSet, valSet, params):
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
    accs = baseline.evaluate(trainSet)
    trainAcc = accs[1]
    accs = baseline.evaluate(valSet)
    valAcc = accs[1]
    cateAcc = accs[2]
    
    return trainAcc, valAcc, cateAcc


def trainWeightedBaseline(dataloader, trainSet, testSet, valSet, params):

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
    
    accs = baseline.evaluate(trainSet)
    trainAcc = accs[1]
    accs = baseline.evaluate(valSet)
    valAcc = accs[1]
    cateAcc = accs[2]
    return trainAcc, valAcc, cateAcc


def KMeanupdate(encode_record, params):
    centroids, label = cluster.vq.kmeans2(encode_record, 10, minit="points")
    weights = powLawOnLabels(label, 10, params)
    return tf.constant(weights, dtype=tf.float32)


def trainNeighbourWeight(dataloader, trainSet, testSet, valSet, weightUpdateFunc, params):
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

    va = metricAcc.result()
    ca = cateAcc.result()
    metricAcc.reset_states()
    for batch, labels in trainSet:
        prediction, _ = splitModel(batch, training=False)
        metricAcc.update_state(labels, prediction)
    ta = metricAcc.result()
    return ta, va, ca


if __name__ == "__main__":
    params = option.read()

    trainFuncs = {
        "None": lambda x: None,
        "baseline": trainBaseline,
        "weightedBaseline": trainWeightedBaseline,
        "kmean": lambda dl, train, test, val, p: trainNeighbourWeight(dl, train, test, val, KMeanupdate, p)
    }
    if params.useGPU:
        utils.selectDevice(0)
    dataloader = data.DataLoader(params)
    avgAcc(params.numTrails, dataloader, trainFuncs[params.trainModel], params)
