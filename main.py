import FilesReader
import numpy as np
import math
import NeuralNetwork
import copy
import sys
import backpropagation as bp
import  verificacao_numerica as vn

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_vetor = np.vectorize(sigmoid)

def exemplo_back_um(layers,lamb, theta_matrices):
    # 3 camadas [1 2 1]

    network = np.array(layers)

    theta1 = theta_matrices[0]

    theta2 = theta_matrices[1]

    thetas = np.array([theta1,theta2])

    regularizacao = lamb
    learning_rate = 1

    entradas = [[0.13], [0.42]]
    saidas = [[0.9], [0.23]]

    exemplos = []
    for i in range(0,2):
        exemplos.append([entradas[i], saidas[i]])

    novos_thetas_back, gradientes_back = bp.backpropagation(exemplos, thetas, regularizacao, network, learning_rate, debug=1)
    novos_thetas_numer, gradientes_numer = vn.numerical_verification(0.00000010000, thetas, exemplos, regularizacao, network,learning_rate, debug=1)

    vn.diff_gradients(gradientes_back, gradientes_numer)

def exemplo_back_two(layers,lamb,theta_matrices):
    # 4 camadas [2 4 3 2]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    theta2 = theta_matrices[1]
    theta3 = theta_matrices[2]

    thetas = np.array([theta1,theta2,theta3])

    regularizacao = lamb
    learning_rate = 1

    entradas = [[0.32, 0.68], [0.83, 0.02]]
    saidas = [[0.75, 0.98], [0.75, 0.28]]

    exemplos = []
    for i in range(0,2):
        exemplos.append([entradas[i], saidas[i]])

    novos_thetas_back, gradientes_back = bp.backpropagation(exemplos, thetas, regularizacao, network, learning_rate, debug=1)
    novos_thetas_numer, gradientes_numer = vn.numerical_verification(0.00000010000, thetas, exemplos, regularizacao, network,
                                                                  learning_rate, debug=1)
    vn.diff_gradients(gradientes_back, gradientes_numer)

def exemplos():
    print("*********** Exemplo Backpropagation 1 *********** ")

    lamb, layers = FilesReader.read_networks("network.txt")
    thetas = FilesReader.read_thetas("initial_weights.txt")
    exemplo_back_um(layers,lamb,thetas)
    #NeuralNetwork.neural_network(layers, lamb, thetas, [[0.13], [0.42]], [[0.9], [0.23]])
    print("")
    print("*********** Exemplo Backpropagation 2 *********** ")

    lamb, layers = FilesReader.read_networks("network2.txt")
    thetas = FilesReader.read_thetas("initial_weights2.txt")
    exemplo_back_two(layers, lamb,thetas)

def salva_resultados(dataset_file, thetas):
    nome_arquivo = "resultado_" + dataset_file
    str_arquivo = ""
    for camada in thetas:
        for line in camada:
            for elemento in line:
                str_arquivo +=  str(round(elemento,5)) + ", "
            str_arquivo = str_arquivo[:-2] + '; '
        str_arquivo = str_arquivo[:-2] + '\n'

    str_arquivo = str_arquivo[:-2]
    #print(str_arquivo)
    f = open(nome_arquivo, "w")
    f.write(str_arquivo)
    f.close()

if __name__ == '__main__':

    if len(sys.argv) != 4:
       exemplos()
    else:
        # python3 main.py network_teste1.txt wine_initial_weights.txt wine.data
        # python3 main.py network_teste1.txt resultado_backpropagation.txt dataset_teste1
        network_file = sys.argv[1]
        weights_file = sys.argv[2]
        dataset_file = sys.argv[3]

        lamb, layers = FilesReader.read_networks(network_file)
        thetas = FilesReader.read_thetas(weights_file)
        instancias = FilesReader.read_dataset_vectorization(dataset_file)

        novos_thetas, gradientes =  NeuralNetwork.neural_network(layers, lamb, thetas, instancias[0],instancias[1])

        salva_resultados(dataset_file, novos_thetas)

