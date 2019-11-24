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

def diff_gradients(gradientes_back, gradientes_numer, debug = 0):
    str_saida = ""
    '''
    for i in range(0, len(gradientes_numer)):
        error_test = 0
        cont = 0
        for j in range(0, len(gradientes_numer[i])):
            for k in range(0, len(gradientes_numer[i][j])):
                test = gradientes_numer[i][j][k]
                test2 = gradientes_back[i][j][k]
                diff = test - test2
                error_test = error_test + abs(diff)
                cont +=1
        error_test = error_test / cont
        str_atual = "Erro entre gradiente via backprop e gradiente numerico para Theta" + str(i + 1) + ": " + str(error_test)
        str_saida += str_atual + "\n"
        if debug == 1:
            print(str_atual)

    '''
    for i in range(0, len(gradientes_back)):
        dist_ab = np.linalg.norm(gradientes_numer[i] - gradientes_back[i])
        size_a = np.linalg.norm(gradientes_back[i])
        size_b = np.linalg.norm(gradientes_numer[i])

        test = size_a + size_b
        test2 = size_b - size_a

        error = dist_ab / (size_a + size_b)
        str_atual = "Erro entre gradiente via backprop e gradiente numerico para Theta" + str(i + 1) + ": " + str(error)
        str_saida +=  str_atual + "\n"
        if debug == 1:
            print(str_atual)
    return str_saida[:-1]


#epsilon=0.0000010000
def numerical_verification(epsilon, thetas, exemplos, regularizacao, network, learning_rate, debug = 0):
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

    if debug == 1:
        print("Rodando verificacao numerica de gradientes (epsilon=0.0000010000)")
        print("")
    cont = 1
    for theta in dv_thetas:
        if debug == 1:
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

    novos_thetas_back, gradientes_back = bp.backpropagation(exemplos, thetas, regularizacao, network, learning_rate,debug=1)
    novos_thetas_numer, gradientes_numer = numerical_verification(0.00000010000, thetas, exemplos, regularizacao, network,learning_rate, debug=1)

    print("")
    print("Gradientes Backpropagation: ")
    print(gradientes_back)
    print("Gradientes Numerical Approximation: ")
    print(gradientes_numer)
    print("")
    str_diferenca = diff_gradients(gradientes_back, gradientes_numer, debug=1)

    return novos_thetas_numer, gradientes_numer, str_diferenca


def escreve_novos_thetas(dataset_file, lamb, thetas, gradientes, str_diferenca):
    nome_arquivo =  "../results/" + "resultado_verificacao_numerica.txt"
    str_arquivo = ""

    str_arquivo += "Dataset: " + dataset_file + "\n"
    str_arquivo += "Fator de Regularização: " + str(lamb) + "\n"
    str_arquivo += "\n"
    str_arquivo += "Novos Thetas" + "\n"
    for camada in thetas:
        for line in camada:
            for elemento in line:
                str_arquivo +=  str(round(elemento,5)) + ", "
            str_arquivo = str_arquivo[:-2] + '; '
        str_arquivo = str_arquivo[:-2] + '\n'

    str_arquivo += "Gradientes" + "\n"

    for camada in gradientes:
        for line in camada:
            for elemento in line:
                str_arquivo += str(round(elemento, 5)) + ", "
            str_arquivo = str_arquivo[:-2] + '; '
        str_arquivo = str_arquivo[:-2] + '\n'

    str_arquivo += str_diferenca

    #str_arquivo = str_arquivo[:-2]
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
        novos_thetas, gradientes, str_diferenca = exemplo_back(layers, lamb, thetas,instancias)

        escreve_novos_thetas(dataset_file,lamb, novos_thetas, gradientes, str_diferenca)