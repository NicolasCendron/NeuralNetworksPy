import FilesReader
import numpy as np
import math
import NeuralNetwork
import sys
import copy
import backpropagation as bp

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_vetor = np.vectorize(sigmoid)

def diff_gradients(gradientes_back, gradientes_numer):
    for i in range(0, len(gradientes_back)):
        dist_ab = np.linalg.norm(gradientes_numer[i] - gradientes_back[i])
        size_a = np.linalg.norm(gradientes_back[i])
        size_b = np.linalg.norm(gradientes_numer[i])

        error = dist_ab / (size_a + size_b)
        print("Erro entre gradiente via backprop e gradiente numerico para Theta" + str(i + 1) + ": " + str(error))

#epsilon=0.0000010000
def numerical_verification(epsilon, thetas, exemplos, regularizacao, network, learning_rate):
    #somar cada peso com espilon separadamente e calcular j, substituir pelo valor do approx para o peso
    dv_thetas = copy.deepcopy(thetas)
    for i in range(0, len(thetas)):
        for j in range(0, len(thetas[i])):
            for k in range(0, len(thetas[i][j])):
                copy_thetas = copy.deepcopy(thetas)
                copy_thetas[i][j][k] = thetas[i][j][k] + epsilon
                first_j = NeuralNetwork.calculaJ(exemplos, copy_thetas, regularizacao, network)
                copy_thetas[i][j][k] = thetas[i][j][k] - epsilon
                second_j = NeuralNetwork.calculaJ(exemplos, copy_thetas, regularizacao, network)
                approx = (first_j - second_j) / (2 * epsilon)
                dv_thetas[i][j][k] = approx

    print("Rodando verificacao numerica de gradientes (epsilon=0.0000010000)")
    print("")
    cont = 1
    for theta in dv_thetas:
        print("Gradiente numerico de Theta" + str(cont) + ":")
        print(theta)
        cont += 1

    novos_thetas = copy.deepcopy(thetas)
    novos_thetas = novos_thetas - dv_thetas * learning_rate

    return novos_thetas, dv_thetas

def exemplo_back(layers,lamb, theta_matrices,instancias):
    # 3 camadas [1 2 1]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    theta2 = theta_matrices[1]

    thetas = np.array([theta1,theta2])

    regularizacao = lamb
    learning_rate = 1
    exemplos = instancias

    novos_thetas_back, gradientes_back = bp.backpropagation(exemplos, thetas, regularizacao, network, learning_rate)
    novos_thetas_numer, gradientes_numer = numerical_verification(0.00000010000, thetas, exemplos, regularizacao, network,learning_rate)
    print("")
    print("Gradientes Backpropagation: ")
    print(gradientes_back)
    print("Gradientes Numerical Approximation: ")
    print(gradientes_numer)
    print("")
    diff_gradients(gradientes_back, gradientes_numer)

    return novos_thetas_numer

def escreve_novos_thetas(thetas):
    print("")
    print("Thetas gerados através da Verificação Numérica:")
    print(thetas)
    nome_arquivo = "resultado_verificacao_numerica.txt"
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
        print(sys.argv)
        print("Wrong number of parameters.")
    else:
        #python3 verificacao_numerica.py network_teste1.txt initial_weights_teste1.txt dataset_teste1
        #python3 verificacao_numerica.py network_teste1.txt resultado_backpropagation.txt dataset_teste1
        network_file = sys.argv[1]
        weights_file = sys.argv[2]
        dataset_file = sys.argv[3]

        lamb, layers = FilesReader.read_networks(network_file)
        thetas = FilesReader.read_thetas(weights_file)
        instancias = FilesReader.read_simple_dataset(dataset_file)
        novos_thetas = exemplo_back(layers, lamb, thetas,instancias)

        escreve_novos_thetas(novos_thetas)
