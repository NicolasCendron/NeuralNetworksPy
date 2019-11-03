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
        self.layer = 0


#1. Inicializar pesos com valores aleatórios não-zero
#b. Calcular o valor de Delta para os neurônios da camada de saída
#c. Calcular o valor de Delta para todos os neurônios nas camadas ocultas
#d. Calcular todos os gradientes
#e. Ajustar todos os pesos da rede pela regra do gradiente descendente
#3. Avaliar performance J da rede no conjunto de treinamento. Se suficientemente boa, ou se melhoria mínima não atingida, parar

#2. Para cada exemplo (x, y) no conjunto de treinamento:
#a. Propagar o exemplo pela rede, calculando sua saída fθ(x)
def compute_neuron_value(neuron):
    x_value = 0

    for n in neuron.input_values:
        x_value += (n.x_value * n.weight)

    return 1/(1 + math.exp(-x_value))




