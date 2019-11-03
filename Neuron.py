import math

class NeuronWeight:

    def __init__(self, x_value, weight):
        self.x_value = x_value
        self.weight = weight


class Neuron:

    def __init__(self, neuron_id, x_value, input_values, output_values, layer):
        self.neuron_id = neuron_id
        self.x_value = x_value
        self.input_values = input_values #list of NeuronWeight
        self.output_values = output_values
        self.layer = layer

    def __init__(self):
        self.neuron_id = "NONE"
        self.x_value = 0
        self.input_values = []
        self.output_values = []
        self.layer = 1


def compute_neuron_value(neuron):
    x_value = 0

    for n in neuron.input_values:
        x_value += (n.x_value * n.weight)

    return 1/(1 + math.exp(-x_value))

