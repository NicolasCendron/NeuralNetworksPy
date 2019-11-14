import FilesReader
import numpy as np
import math
import NeuralNetwork


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_vetor = np.vectorize(sigmoid)


def propagacao(exemplo,thetas,network):
    entrada = list(exemplo[0])
    print("propagando entrada" + str(entrada))

    ativacao = []
    ativacao.append([1] + entrada)
    print("ativacao 1")
    print(ativacao[0])
    Z = []
    for i in range(1,len(network) - 1):
        Zatual = (thetas[i-1]).dot(ativacao[i-1])
        print("z" + str(i + 1))
        print(Zatual)
        Z.append(Zatual)
        ativacaoAtual = np.insert(sigmoid_vetor(Zatual), 0, 1)
        ativacao.append(ativacaoAtual)
        print("a" + str(i + 1))
        print(ativacao[-1])

    ZFinal = thetas[-1].dot(ativacao[-1])
    ativacao_final = sigmoid_vetor(ZFinal)

    print("ZFinal")
    print(ZFinal)
    print("AFinal")
    print(ativacao_final)
    return ativacao_final


def calculaJ(exemplos, thetas, regularizacao, network, learning_rate):
    J = 0
    cont = 0
    # # print('network')
    # # print(network)

    D = []
    for i in range(len(network)):
        D.append([])

    for exemplo in exemplos:
        cont += 1
        # print("Processando exemplo de treinamento " + str(cont))
        entradas = np.array(exemplo[0])
        saidas = np.array(exemplo[1])

        # 1.1 Propaga x(i) e obtém as saídas f(x(i)) preditas pela rede
        ativacao, saidas_preditas = propagacao(exemplo, thetas, network)

        # 1.2 calcula deltas para os neurônios da camada de saída
        error = saidas_preditas - saidas

        # cria array para armazenar os deltas
        deltas = []
        for i in range(len(network)):
            deltas.append([])
        deltas[-1] = error
        # print('ativacao', len(ativacao))

        # 1.3 Para cada camada k=L-1…2, calcula os deltas para as camadas ocultas
        for k in reversed(range(1, len(network)-1)):
            delta = np.multiply(np.multiply(np.transpose(thetas[k]) * deltas[-1], ativacao[k]), (1-ativacao[k]))  #.* ativacao[k] .* (1-ativacao[k])
            delta_without_bias = []

            # Remove o primeiro elemento de delta(l=k) (i.e., o delta associado ao neurônio de bias da camada k
            for i in range(len(delta)):
                delta_without_bias.append(np.delete(delta[i], 0, None))
            deltas[k] = delta_without_bias
        
        # 1.4 Para cada camada k=L-1…1, atualiza os gradientes dos pesos de cada camada com base no exemplo atua
        for k in reversed(range(0, len(network)-1)):
            factor = deltas[k+1] * np.transpose(ativacao[k])
            if len(D[k]) == 0:
                D[k] = factor
            else:
                D[k] += factor
            # print('D[', k, D)

        # 2. Calcula gradientes finais (regularizados) para os pesos de cada camada
        for k in reversed(range(0, len(network)-1)):
            # 2.1 aplica regularização λ apenas a pesos não bias
            thetas[k].insert(0, 0)
            thetas[k] = geek.insert(arr, 1, 9) 
            Pk = np.multiply(regularizacao, thetas[k])
            # 2.2 combina gradientes com regularização; divide pelo num de exemplos para calcular gradiente médio
            # print('D[' + str(k) + ']')
            # print(D[k])
            # print('P[' + str(k) + ']')
            # print(Pk)
            print(thetas[k])
            D[k] = (1/len(exemplos)) * (D[k] + Pk)

        #4. atualiza pesos de cada camada com base nos gradientes
        for k in reversed(range(0, len(network)-1)):
            thetas[k] = thetas[k] - np.multiply(learning_rate, D[k])

        vetorJ = (np.negative(saidas)).dot( np.log(saidas_preditas))
        vetorJ -= (np.ones(saidas.size) - saidas).dot(np.log(np.ones(saidas_preditas.size) - saidas_preditas))
        J += np.sum(vetorJ)

    J = J/len(exemplos)
    S = 0
    for theta_vector in thetas[0]:
        for theta in theta_vector:
            S+= math.sqrt(theta)
    S = regularizacao/(2*len(entradas))*S

    # print("J para o total do dataset")
    # print(str(J + S))
    return J + S

def exemplo_back_um(layers,lamb, theta_matrices):
    # 3 camadas [1 2 1]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    #theta1 = np.array([ [0.4 , 0.1] ,
    #                    [0.3 , 0.2]])

    theta2 = theta_matrices[1]
    #theta2 = np.array([0.7, 0.5, 0.6])

    thetas = np.array([theta1,theta2])

    regularizacao = lamb

    entradas = [[0.13], [0.42]]
    saidas = [[0.9], [0.23]]

    exemplos = []
    for i in range(0,2):
        exemplos.append([entradas[i], saidas[i]])

    print(exemplos)
    Jexemplo = calculaJ(exemplos, thetas, regularizacao, network, learning_rate)

def exemplo_back_two(layers,lamb,theta_matrices):
    # 4 camadas [2 4 3 2]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    theta2 = theta_matrices[1]
    theta3 = theta_matrices[2]

    thetas = np.array([theta1,theta2,theta3])

    regularizacao = lamb
    learning_rate = 0.01

    entradas = [[0.32, 0.68], [0.83, 0.02]]
    saidas = [[0.75, 0.98], [0.75, 0.28]]

    exemplos = []
    for i in range(0,2):
        exemplos.append([entradas[i], saidas[i]])

    print(exemplos)
    Jexemplo = calculaJ(exemplos, thetas, regularizacao, network, learning_rate)


if __name__ == '__main__':

    arquivo = "pima.tsv"
    lamb, layers = FilesReader.read_networks("pima_network.txt")
    thetas = FilesReader.read_thetas("pima_initial_weights.txt")

    #arquivo = "wine.data"
    #lamb, layers = FilesReader.read_networks("wine_network.txt")
    #thetas = FilesReader.read_thetas("wine_initial_weights.txt")

    #arquivo = "ionosphere.data"
    #lamb, layers = FilesReader.read_networks("ionosphere_network.txt")
    #thetas = FilesReader.read_thetas("ionosphere_initial_weights.txt")

    inputs, outputs = FilesReader.read_dataset_vectorization(arquivo)
    NeuralNetwork.neural_network(layers,lamb,thetas,inputs, outputs)

    lamb, layers = FilesReader.read_networks("network.txt")
    thetas = FilesReader.read_thetas("initial_weights.txt")
    #exemplo_back_um(layers,lamb,thetas)
    NeuralNetwork.neural_network(layers, lamb, thetas, [[0.13], [0.42]], [[0.9], [0.23]])

    lamb, layers = FilesReader.read_networks("network2.txt")
    thetas = FilesReader.read_thetas("initial_weights2.txt")
    #exemplo_back_two(layers, lamb,thetas)
    NeuralNetwork.neural_network(layers, lamb, thetas, [[0.32, 0.68], [0.83, 0.02]], [[0.75, 0.98], [0.75, 0.28]])
