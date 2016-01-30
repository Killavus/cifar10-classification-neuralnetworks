from load_data import LoadDataStreams, BatchSize
from network_utils import BuildNeuralNetwork 
import time

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *

from functools import partial

NETWORK_SPEC = [
  { 'type': Conv2DLayer,
    'args': {
      'num_filters': 32,
      'filter_size': (5, 5),
      'nonlinearity': rectify,
      'W': lasagne.init.GlorotUniform()
    }
  },
  { 'type': MaxPool2DLayer,
    'args': {
      'pool_size': (2, 2)
    }
  },
  { 'type': Conv2DLayer,
    'args': {
      'num_filters': 32,
      'filter_size': (5, 5),
      'nonlinearity': rectify,
      'W': lasagne.init.GlorotUniform()
    }
  },
  { 'type': MaxPool2DLayer,
    'args': {
      'pool_size': (2, 2)
    }
  },
  { 'type': dropout,
    'args': { 'p': 0.5 }
  },
  { 'type': DenseLayer,
    'args': {
      'num_units': 200,
      'nonlinearity': rectify
    }
  },
  { 'type': DenseLayer,
    'args': {
      'num_units': 10,
      'nonlinearity': softmax
    }
  }
]

LEARNING_RATE=0.001
MOMENTUM=0.95

input_values = T.tensor4('inputs')
input_targets = T.ivector('targets')

INPUT_SPEC = {
  'shape': (BatchSize(), 3, 32, 32),
  'input_var': input_values
}

def TrainNeuralNetwork(input_var, targets_var, data, network, iterations=2,
        verify_fns=[], verify_data_col=[]):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, targets_var)
    loss = loss.mean()

    net_params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, net_params,
                                                learning_rate=LEARNING_RATE, 
                                                momentum=MOMENTUM)

    train_fn = theano.function([input_var, targets_var], loss, updates=updates)

    for it in range(iterations):
        try:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            print "Iteration %d..." % (it+1)
            
            for inputs, targets in data.get_epoch_iterator():
                targets = targets.T.ravel()
                train_err += train_fn(inputs, targets)
                train_batches += 1

            print "Iteration %d completed. Loss: %.6f" % (it+1, 
                    train_err / train_batches)

            for index, verify_fn in enumerate(verify_fns):
                verify_fn(input_var, targets_var, verify_data_col[index],
                        network)

            print "It took %.3fs" % (time.time() - start_time)
        except KeyboardInterrupt:
            return lasagne.layers.get_all_param_values(network)

    return lasagne.layers.get_all_param_values(network)

def VerifyNetworkAccuracy(name, input_var, targets_var, data, network):
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            targets_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), targets_var),
                           dtype=theano.config.floatX)

    val_batches = 0
    val_err = 0
    val_acc = 0
    val_fn = theano.function([input_var, targets_var], [test_loss, test_acc])

    for inputs, targets in data.get_epoch_iterator():
        targets = targets.T.ravel()
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    print "%s loss: %.6f" % (name, val_err / val_batches)
    print "%s accuracy: %.2f %s" % (name, val_acc / val_batches * 100, "%")

network = BuildNeuralNetwork(NETWORK_SPEC, INPUT_SPEC)

train_stream, validation_stream, test_stream = LoadDataStreams()

TrainNeuralNetwork(input_values, input_targets, train_stream, network, 100,
                   [partial(VerifyNetworkAccuracy, "Validation"), 
                    partial(VerifyNetworkAccuracy, "Test")],
                   [validation_stream, test_stream])
