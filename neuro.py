import numpy as np


def relu(value):
    return max(0, value)


def add_bias(x):
    return np.concatenate([x, np.matrix([[1]])])


class NeuroBasket:
    net_structure = []
    weight_layers = []
    relu_activation = np.vectorize(relu)

    def __init__(self, structure=[]):
        self.net_structure = structure

    def populateNet(self):
        layers_count = len(self.net_structure) - 1
        self.weight_layers = []

        for i in range(0, layers_count):
            columns = self.net_structure[i] + 1
            rows = self.net_structure[i + 1]

            element_count = rows * columns
            elements = np.random.uniform(-1.5, 1.5, element_count)

            weight_layer = np.matrix([elements]).reshape((rows, columns))
            self.weight_layers.append(weight_layer)

    def run(self, input):
        layer_count = len(self.weight_layers)
        input_layer = add_bias(input)

        last_layer = input_layer

        for i in range(0, layer_count):
            last_layer = self.weight_layers[i] * last_layer
            if is_not_last_layer(i, layer_count):
                last_layer = self.relu_activation(add_bias(last_layer))

        return last_layer


def multiple_instances(net_structure, size):
    nets = []

    for x in range(0, size):

        net = NeuroBasket(net_structure)
        net.populateNet()
        nets.append(net)

    return nets


def is_not_last_layer(i, layer_count):
    return True if i != layer_count - 1 else False
