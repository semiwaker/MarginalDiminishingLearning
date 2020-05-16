import os.path as path

import tensorflow as tf
import tensorflow.keras as keras

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


if __name__ == "__main__":
    params = option.read()

    trainFuncs = {
        "None": lambda x: None,
        "baseline": trainBaseline
    }
    if params.useGPU:
        utils.selectDevice(0)
    trainFuncs[params.trainModel](params)
