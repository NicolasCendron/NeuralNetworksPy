import FilesReader
import numpy as np
import math
import NeuralNetwork
import sys

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_vetor = np.vectorize(sigmoid)

def backpropagation(exemplos, thetas, regularizacao, network, learning_rate, debug = 0):
    J = 0
    cont = 0
    gradientes = []
    D = []
    for i in range(len(network) - 1):
        D.append([])

    for exemplo in exemplos:
        cont += 1
        if debug == 1:
            print("Calculando gradientes com base no exemplo  " + str(cont))
        #entradas = np.array(exemplo[0])
        saidas = np.array(exemplo[1])

        # 1.1 Propaga x(i) e obtém as saídas f(x(i)) preditas pela rede
        ativacao, saidas_preditas = NeuralNetwork.propagation(exemplo, thetas, network)

        # 1.2 calcula deltas para os neurônios da camada de saída
        error = saidas_preditas - saidas

        # cria array para armazenar os deltas
        deltas = []
        for i in range(len(network)):
            deltas.append([])
        deltas[-1] = error
        if debug == 1:
            print("delta"+str(len(network)))
            print(error)

        # 1.3 Para cada camada k=L-1…2, calcula os deltas para as camadas ocultas
        for k in reversed(range(1, len(network)-1)):
            delta = np.transpose(thetas[k]).dot(deltas[k+1])  #.* ativacao[k] .* (1-ativacao[k])
            delta = np.multiply(delta, ativacao[k])
            delta = np.multiply(delta, (1 - ativacao[k]))

            #; Remove o primeiro elemento de delta(l=k) (i.e., o delta associado ao neurônio de bias da camada k
            deltas[k] = delta[1:]
            if debug == 1:
                print("delta"+str(k+1))
                print(deltas[k])
        # 1.4 Para cada camada k=L-1…1, atualiza os gradientes dos pesos de cada camada com base no exemplo atual

        for k in reversed(range(0, len(network)-1)):
            factor = []

            for a in deltas[k+1]:
                line = []
                for b in ativacao[k]:
                    line.append(a*b)

                factor.append(line)

            factor = np.array(factor)
            #factor = deltas[k+1] * np.transpose(ativacao[k]) #não funciona por alguma razão
            if debug == 1:
                print("Gradientes de Theta" + str(k + 1) + " com base no exemplo" + str(cont))
                print(factor)
            if len(D[k]) == 0:
                D[k] = factor
            else:
               D[k] += factor
#
    # 2. Calcula gradientes finais (regularizados) para os pesos de cada camada
    for k in reversed(range(0, len(network)-1)):
        # 2.1 aplica regularização λ apenas a pesos não bias

        theta_bias_zerado = np.copy(thetas[k])
        for line in theta_bias_zerado:
            line[0] = 0
        Pk = np.multiply(regularizacao, theta_bias_zerado)
        # 2.2 combina gradientes com regularização; divide pelo num de exemplos para calcular gradiente médio

        D[k] = (1/len(exemplos)) * (D[k] + Pk)

    #4. atualiza pesos de cada camada com base nos gradientes
    if debug == 1:
        print("Dataset completo processado. Calculando gradientes regularizados")

    novos_thetas = np.copy(thetas)
    for k in range(0, len(network)-1):

        gradiente = np.multiply(learning_rate, D[k])
        if debug == 1:
            print("Gradientes finais para Theta" + str(k+1) + " (com regularizacao):")
            print(gradiente)
        gradientes.append(gradiente)
        novos_thetas[k] = thetas[k] - gradiente

    return novos_thetas, gradientes


def exemplo_back(layers,lamb, theta_matrices,instancias):

    network = np.array(layers)

    theta1 = theta_matrices[0]

    theta2 = theta_matrices[1]

    thetas = np.array([theta1,theta2])

    regularizacao = lamb
    learning_rate = 1

    exemplos = instancias

    novos_thetas, gradientes = backpropagation(exemplos, thetas, regularizacao, network, learning_rate, debug=1)
    return novos_thetas, gradientes

def escreve_novos_thetas(dataset_file,lamb, thetas, gradientes):
    print(thetas)
    nome_arquivo = "../results/" + "resultado_backpropagation.txt"
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
        #python3 backpropagation.py network_teste1.txt initial_weights_teste1.txt dataset_teste1
        #python3 backpropagation.py network_teste1.txt resultado_backpropagation.txt dataset_teste1
        network_file = sys.argv[1]
        weights_file = sys.argv[2]
        dataset_file = sys.argv[3]

        lamb, layers = FilesReader.read_networks(network_file)
        thetas = FilesReader.read_thetas(weights_file)
        instancias = FilesReader.read_simple_dataset(dataset_file)
        novos_thetas, gradientes = exemplo_back(layers, lamb, thetas,instancias)

        escreve_novos_thetas(dataset_file,lamb, novos_thetas, gradientes)

        #lamb, layers = FilesReader.read_networks("network.txt")
        #thetas = FilesReader.read_thetas("initial_weights.txt")
        #exemplo_back_um(layers,lamb,thetas)
        #NeuralNetwork.neural_network(layers, lamb, thetas, [[0.13], [0.42]], [[0.9], [0.23]])

