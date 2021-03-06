import os
import os.path as path
import sys
import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import scipy.cluster as cluster
import scipy.stats as stats
from sklearn.neighbors import KernelDensity, BallTree

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
    label_weight *= params.beta
    weights = np.array([label_weight[i] for i in labels])
    print(np.max(weights), np.min(weights))
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
    print(f"Finished {timer()} {sys.argv}")
    print("Overall train accuracy %.4f" % (float(trainAcc)))
    print("Overall val accuracy %.4f" % (float(valAcc)))
    print("Overall cate accuracy ", end='')
    for i in cateAcc[:-1]:
        print("%.2lf," % i, end=' ')
    print("%.3f" % cateAcc[-1])


def trainBaseline(dataloader, trainSet, testSet, valSet, params):
    baseline = model.makeBaselineModel(params)

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()
    baseline.compile(optimizer, loss=loss_fn, metrics=[metricAcc, cateAcc])
    
    baseline.fit(trainSet, epochs=params.numEpochs, validation_data=testSet)
    
    baseline.save_weights(path.join(params.modelPath, "baseline.keras"))

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


def KMeanDistupdate(encode_record, params):
    centroids, label = cluster.vq.kmeans2(encode_record, 10, minit="points")
    weights = powLawOnLabels(label, 10, params)
    dist = np.array([np.linalg.norm(centroids[l]-encode_record[i])
                     for i, l in enumerate(label)])
    dist = np.power(dist * 0.1, -params.alpha)
    weights *= dist
    print(np.max(weights), np.min(weights))
    return tf.constant(weights, dtype=tf.float32)


def radiusUpdate(encode_record, params):
    encode_record -= np.mean(encode_record, 0)
    print(np.max(np.max(encode_record)), np.min(np.min(encode_record)))
    tree = BallTree(encode_record)
    neighbor = tree.query_radius(encode_record, 3, count_only=True) + 1
    print(np.max(neighbor), np.min(neighbor))
    weights = np.power(neighbor, params.alpha) * params.beta
    return tf.constant(weights, dtype=tf.float32)


def normalize(weight):
    return weight / np.mean(weight)


def Onlylossupdate(old_loss, loss, params):
    loss_weight = np.array(loss)
    # loss_weight = normalize(loss_weight)
    loss_weight = np.power(loss_weight, params.alpha)
    loss_weight = normalize(loss_weight)
    return tf.constant(loss_weight, dtype=tf.float32)


def Trendlossupdate(old_loss, loss, params):
    loss_weight = np.array(loss)
    old_loss_weight = np.array(old_loss)
    trend_weight = np.abs(old_loss_weight - loss_weight) / loss_weight
    trend_weight = normalize(trend_weight)
    loss_weight = normalize(loss_weight)
    loss_weight = loss_weight * trend_weight
    loss_weight = normalize(loss_weight)
    loss_weight = np.power(loss_weight, params.alpha)
    loss_weight = normalize(loss_weight)
    return tf.constant(loss_weight, dtype=tf.float32)


def FeatureL1update(old_encode_record, encode_record, params):
    diff = [np.abs(old_encode_record[i] - encode_record[i]) for i in range(len(encode_record))]
    loss_weight = [np.mean(diff[i]) for i in range(len(diff))]
    loss_weight = normalize(loss_weight)
    loss_weight = np.power(loss_weight, params.alpha)
    loss_weight = normalize(loss_weight)
    return tf.constant(loss_weight, dtype=tf.float32)


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


def trainLossWeight(dataloader, trainSet, testSet, valSet, weightUpdateFunc, params):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    splitModel = model.makeSplitModel(params)

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    loss_fn_2 = keras.losses.categorical_crossentropy
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()

    datalen = dataloader.trainLen
    weights = tf.ones([datalen], dtype=tf.float32)
    timer = utils.Timer()
    # trainables = [splitModel.conv.trainable_variables,
    #   splitModel.predict.trainable_variables]
    old_loss_record = []
    for epochID in range(1, params.numEpochs+1):
        IDcnt = 0
        batchCnt = 0
        metricAcc.reset_states()
        trainloss = 0
        # encode_record = []
        loss_record = []
        for batch, labels in trainSet:
            batchSize = batch.shape[0]
            batchWeight = weights[IDcnt: IDcnt+batchSize]
            with tf.GradientTape() as tape:
                prediction, encoding = splitModel(batch, training=True)
                loss = loss_fn(labels, prediction, batchWeight)
                loss_vec = loss_fn_2(labels, prediction)
            trainloss += loss
            gradient = tape.gradient(loss, splitModel.trainable_variables)
            optimizer.apply_gradients(
                zip(gradient, splitModel.trainable_variables))
            metricAcc.update_state(labels, prediction)

            # encode_record.append(encoding.numpy())
            loss_record.append(loss_vec.numpy())

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
            loss_record = np.concatenate(loss_record)
            weights = weightUpdateFunc(old_loss_record, loss_record, params)
            print(weights)
            old_loss_record = loss_record
        else:
            old_loss_record = np.concatenate(loss_record)

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


def trainFeatureWeight(dataloader, trainSet, testSet, valSet, weightUpdateFunc, params):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    splitModel = model.makeSplitModel(params)

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    loss_fn_2 = keras.losses.categorical_crossentropy
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
        # loss_record = []
        for batch, labels in trainSet:
            batchSize = batch.shape[0]
            batchWeight = weights[IDcnt: IDcnt+batchSize]
            with tf.GradientTape() as tape:
                prediction, encoding = splitModel(batch, training=True)
                loss = loss_fn(labels, prediction, batchWeight)
                # loss_vec = loss_fn_2(labels, prediction)
            trainloss += loss
            gradient = tape.gradient(loss, splitModel.trainable_variables)
            optimizer.apply_gradients(
                zip(gradient, splitModel.trainable_variables))
            metricAcc.update_state(labels, prediction)

            encode_record.append(encoding.numpy())
            # loss_record.append(loss_vec.numpy())

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
            weights = weightUpdateFunc(old_encode_record, encode_record, params)
            print(weights)
            old_encode_record = encode_record
        else:
            old_encode_record = np.concatenate(encode_record)

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


def trainKmeanBatch(dataloader, trainSet, testSet, valSet, params, withDist=False):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    splitModel = model.makeSplitModel(params)

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()

    datalen = dataloader.trainLen
    timer = utils.Timer()
    centroids = []
    clusterWeight = np.ones([10])
    # trainables = [splitModel.conv.trainable_variables,
    #   splitModel.predict.trainable_variables]
    for epochID in range(1, params.numEpochs+1):
        IDcnt = 0
        batchCnt = 0
        metricAcc.reset_states()
        trainloss = 0
        encode_record = []
        maxWeight = np.max(clusterWeight)
        minWeight = np.min(clusterWeight)
        if withDist:
            maxWeight = 0
            minWeight = 10000
        for batch, labels in trainSet:
            batchSize = batch.shape[0]
            with tf.GradientTape() as tape:
                prediction, encoding = splitModel(batch, training=True)
                if epochID <= params.warmUpEpochs:
                    batchWeight = 1.0
                else:
                    encode = tf.reduce_mean(encoding, 0)
                    dists = np.array([np.linalg.norm(encode-c)
                                      for c in centroids])
                    idx = np.argmin(dists)
                    batchWeight = clusterWeight[idx]
                loss = loss_fn(labels, prediction)
                loss *= batchWeight
                # print(loss)
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
        print(f"Weight: [{minWeight}, {maxWeight}]")
        print(f"Train Loss: {trainloss} Accuracy: {trainAcc}")
        print(f"Test Loss: {testloss} Accuracy: {testAcc}")

        if epochID >= params.warmUpEpochs:
            encode_record = np.concatenate(encode_record)
            centroids, label = cluster.vq.kmeans2(
                encode_record, 10, minit="points")
            cnt = [0] * 10
            for i in label:
                cnt[i] += 1
            clusterWeight = np.array(cnt)
            clusterWeight = np.power(clusterWeight, params.alpha)
            clusterWeight /= np.sum(clusterWeight)
            clusterWeight *= params.beta

    splitModel.save_weights(
        path.join(params.modelPath, "splitModel.keras"))
    metricAcc.reset_states()
    cateAcc.reset_states()
    for batch, labels in valSet:
        prediction, _ = splitModel(batch, training=False)
        metricAcc.update_state(labels, prediction)
        cateAcc.update_state(labels, prediction)
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
        "kmean": lambda dl, train, test, val, p: trainNeighbourWeight(dl, train, test, val, KMeanupdate, p),
        "kmeanDist": lambda dl, train, test, val, p: trainNeighbourWeight(dl, train, test, val, KMeanDistupdate, p),
        "naiveloss": lambda dl, train, test, val, p: trainLossWeight(dl, train, test, val, Onlylossupdate, p),
        "trendloss": lambda dl, train, test, val, p: trainLossWeight(dl, train, test, val, Trendlossupdate, p),
        "featureL1": lambda dl, train, test, val, p: trainFeatureWeight(dl, train, test, val, FeatureL1update, p),
        "kmeanBatch": lambda dl, train, test, val, p: trainKmeanBatch(dl, train, test, val, p),
        "radius": lambda dl, train, test, val, p: trainNeighbourWeight(dl, train, test, val, radiusUpdate, p)
    }
    if params.useGPU:
        utils.selectDevice(0)
    dataloader = data.DataLoader(params)
    avgAcc(params.numTrials, dataloader, trainFuncs[params.trainModel], params)
