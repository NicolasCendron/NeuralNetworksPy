import FilesReader
import numpy as np
import math
import NeuralNetwork
import copy
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

    novos_thetas_back, gradientes_back = bp.backpropagation(exemplos, thetas, regularizacao, network, learning_rate)
    novos_thetas_numer, gradientes_numer = numerical_verification(0.00000010000, thetas, exemplos, regularizacao, network,learning_rate)

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

    novos_thetas_back, gradientes_back = bp.backpropagation(exemplos, thetas, regularizacao, network, learning_rate)
    novos_thetas_numer, gradientes_numer = vn.numerical_verification(0.00000010000, thetas, exemplos, regularizacao, network,
                                                                  learning_rate)
    vn.diff_gradients(gradientes_back, gradientes_numer)

if __name__ == '__main__':


    #lamb, layers = FilesReader.read_networks("network.txt")
    #thetas = FilesReader.read_thetas("initial_weights.txt")
    #exemplo_back_um(layers,lamb,thetas)
    #NeuralNetwork.neural_network(layers, lamb, thetas, [[0.13], [0.42]], [[0.9], [0.23]])

    lamb, layers = FilesReader.read_networks("network2.txt")
    thetas = FilesReader.read_thetas("initial_weights2.txt")
    exemplo_back_two(layers, lamb,thetas)
    #NeuralNetwork.neural_network(layers, lamb, thetas, [[0.32, 0.68], [0.83, 0.02]], [[0.75, 0.98], [0.75, 0.28]])
