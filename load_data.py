from fuel.datasets.cifar10 import CIFAR10
from fuel.transformers import ScaleAndShift, Cast
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme

import numpy as np

BATCH_SIZE = 100

def BatchSize():
    return BATCH_SIZE

CIFAR10.default_transformers = (
  (ScaleAndShift, [2.0 / 255.0, -1], {'which_sources': 'features'}),
  (Cast, [np.float32], {'which_sources': 'features'}),
)

def LoadDataStreams():
    train = CIFAR10(("train",), subset=slice(None, 40000))
    validation = CIFAR10(("train",), subset=slice(40000, None))
    test = CIFAR10(("test",))

    train_stream = DataStream.default_stream(train, 
                                             iteration_scheme=ShuffledScheme(
                                                 train.num_examples,
                                                 BATCH_SIZE))

    test_stream = DataStream.default_stream(test, 
                                            iteration_scheme=SequentialScheme(
                                                test.num_examples,
                                                BATCH_SIZE))
                                             
    validation_stream = DataStream.default_stream(validation, 
                                                  iteration_scheme=SequentialScheme(
                                                  validation.num_examples,
                                                  BATCH_SIZE))
    return train_stream, validation_stream, test_stream
