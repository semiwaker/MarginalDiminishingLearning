import argparse


def read():
    parser = argparse.ArgumentParser(description='MarginalDiminishingLearning')

    parser.add_argument('--useGPU', action="store_true",
                        default=False, help="Use GPU")
    
    parser.add_argument('--dataUnbias', action="store_true",
                        default=False, help="Stop data bias")
    parser.add_argument('--numTrials', default=10, type=int,
                        help="Number of trials")

    parser.add_argument('--trainModel', default="None",
                        type=str, help="Specify which model to train")
    parser.add_argument('--continueModel', default="None",
                        type=str, help="Specify which model to continue training")
    
    parser.add_argument('--dataset', default='data/cifar-10-batches-py',
                        type=str, help='Path to the dataset')
    parser.add_argument('--dataBatch', default=1, type=int,
                        help='Specified which data batch to use')
    parser.add_argument('--splitPath', default='data/split.json',
                        type=str, help='Path to the split file')
    parser.add_argument('--modelPath', default='saved_model',
                        type=str, help='Path to the saved model')
    
    parser.add_argument('--batchSize', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--numEpochs', default=30, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--learningRate', default=1e-3, type=float,
                        help='Learning rate')
    
    parser.add_argument('--alpha', default=-1.0, type=float,
                        help="Alpha coefficient in the power law of learning")
    parser.add_argument('--beta', default=1.0, type=float,
                        help="Beta coefficient in the power law of learning")
    parser.add_argument('--warmUpEpochs', default=1, type=int,
                        help="Warm up epochs before calculating weight")
    
    parsed = parser.parse_args()
    
    print("Marginal Diminishing Learning")
    print("Args:")
    for u, v in parsed.__dict__.items():
        print(f"{u} : {v}")
    
    return parsed
