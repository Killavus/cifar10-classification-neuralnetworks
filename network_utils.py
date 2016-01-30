import lasagne

def BuildNeuralNetwork(layer_specification, input_args):
    network_layer = lasagne.layers.InputLayer(**input_args)    

    for specification in layer_specification:
        layer_type = specification.get('type')
        layer_args = specification.get('args')
        network_layer = layer_type(network_layer, **layer_args)

    return network_layer
