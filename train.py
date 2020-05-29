import os.path as path

import tensorflow as tf
import tensorflow.keras as keras
import scipy

import option
import data
import model
import utils


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

    baseline = model.makeBaselineModel(params)
    baseline.summary()

    labels = dataloader.trainData[b"labels"]
    total = len(labels)
    cnt = [0] * 10
    for i in labels:
        cnt[i] += 1
    label_weight = [total / i for i in cnt]
    total = sum(label_weight)
    label_weight = [i / total for i in label_weight]
    weights = [label_weight[i] for i in labels]

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
    print(accs)


def KNNupdate(encode_record, params):
    centroids, label = scipy.cluster.vq.kmean2(encode_record, 10)
    clusterCnt = [0] * 10
    for i in label:
        clusterCnt[i] += 1
    

def trainNeighbourWeight(params, weightUpdateFunc):
    dataloader = data.DataLoader(params)
    trainSet, testSet, valSet = dataloader.readData()

    splitModel = model.SplitModel(params)
    splitModel.summary()
    print(splitModel.trainable_variables)

    optimizer = keras.optimizers.Adam(lr=params.learningRate)
    loss_fn = keras.losses.CategoricalCrossentropy()
    metricAcc = keras.metrics.CategoricalAccuracy()
    cateAcc = model.CateAcc()

    datalen = len(dataloader.trainData)
    weights = tf.ones([datalen], dtype=tf.float32)
    encode_record = tf.TensorArray(tf.float32, size=datalen / params.batchSize + 1)
    timer = utils.Timer()
    for epochID in range(1, params.numEpochs+1):
        IDcnt = 0
        batchCnt = 0
        metricAcc.reset_states()
        trainloss = 0
        for batch, labels in trainSet:
            batchSize = batch.shape[0]
            batchWeight = weights[IDcnt: IDcnt+batchSize]
            with tf.GradientTape() as tape:
                encoding, prediction = splitModel(batch, training=True)
                loss = loss_fn(labels, prediction, batchWeight)
            trainloss += loss
            gradient = tape.gradient(loss, splitModel.trainable_variables)
            optimizer.apply_gradients(
                zip(gradient, splitModel.trainable_variables))
            metricAcc.update_state(labels, prediction)

            encode_record.write(batchCnt, encoding)

            batchCnt += 1
            IDcnt += batchSize
        trainAcc = metricAcc.result()
        trainloss /= batchCnt

        batchCnt = 0
        testloss = 0
        metricAcc.reset_states()
        for batch, labels in testSet:
            _, prediction = splitModel(batch, training=False)
            loss = loss_fn(labels, prediction, batchWeight)
            testloss += loss
            batchCnt += 1
            metricAcc.update_state(labels, prediction)
        testAcc = metricAcc.result()
        testloss /= batchCnt

        print(f"Epoch {epochID} {timer()}")
        print(f"Train Loss: {trainloss} Accuracy: {trainAcc}")
        print(f"Test Loss: {testloss} Accuracy: {testAcc}")

        if epochID >= params.warmUpEpochs:
            encode_record = encode_record.concat()
            weights = weightUpdateFunc(encode_record, params)

    metricAcc.reset_states()
    cateAcc.reset_states()
    for batch, labels in valSet:
        _, prediction = splitModel(batch, training=False)


if __name__ == "__main__":
    params = option.read()

    trainFuncs = {
        "None": lambda x: None,
        "baseline": trainBaseline,
        "weightedBaseline": trainWeightedBaseline
    }
    if params.useGPU:
        utils.selectDevice(0)
    trainFuncs[params.trainModel](params)
