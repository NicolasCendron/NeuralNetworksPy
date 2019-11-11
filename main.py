import FilesReader
import numpy as np
import math



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
        ativacaoAtual = np.insert(sigmoid_vetor(Z), 0, 1)
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


def calculaJ(exemplos,thetas, regularizacao, network):
    J = 0
    cont = 0
    for exemplo in exemplos:
        cont += 1
        print("Processando exemplo de treinamento " + str(cont))

        entradas = np.array(exemplo[0])
        saidas = np.array(exemplo[1])

        saidas_preditas = propagacao(exemplo, thetas, network)

        print("Saidas Preditas para o Exemplo" + str(cont))
        print(saidas_preditas)
        print("Saidas Esperadas para o Exemplo" + str(cont))
        print(saidas)

        vetorJ = (np.negative(saidas)).dot( np.log(saidas_preditas))
        vetorJ -= (np.ones(saidas.size) - saidas).dot(np.log(np.ones(saidas_preditas.size) - saidas_preditas))

        print("J para o Exemplo" + str(cont))
        print(np.sum(vetorJ))

        J += np.sum(vetorJ)

    J = J/ len(exemplos)
    S = 0
    for theta_vector in thetas[0]:
        for theta in theta_vector:
            S+= math.sqrt(theta)
    S = regularizacao/(2*len(entradas))*S

    print("J para o total do dataset")
    print(str(J + S))
    return J + S

def exemplo_back_um(layers,theta_matrices):
    # 3 camadas [1 2 1]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    #theta1 = np.array([ [0.4 , 0.1] ,
    #                    [0.3 , 0.2]])

    theta2 = theta_matrices[1]
    #theta2 = np.array([0.7, 0.5, 0.6])

    thetas = np.array([theta1,theta2])

    regularizacao = 0

    entradas = [[0.13], [0.42]]
    saidas = [[0.9], [0.23]]

    exemplos = []
    for i in range(0,2):
        exemplos.append([entradas[i], saidas[i]])

    print(exemplos)
    Jexemplo = calculaJ(exemplos,thetas,regularizacao,network)

def exemplo_back_two(layers,theta_matrices):
    # 4 camadas [2 4 3 2]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    theta2 = theta_matrices[1]
    theta3 = theta_matrices[2]

    thetas = np.array([theta1,theta2,theta3])

    regularizacao = 0

    entradas = [[0.32, 0.68], [0.83, 0.02]]
    saidas = [[0.75, 0.98], [0.75, 0.28]]

    exemplos = []
    for i in range(0,2):
        exemplos.append([entradas[i], saidas[i]])

    print(exemplos)
    Jexemplo = calculaJ(exemplos,thetas,regularizacao,network)


if __name__ == '__main__':
    arquivo = "pima.tsv"
    lamb, layers = FilesReader.read_networks("network.txt")
    thetas = FilesReader.read_thetas("initial_weights.txt")
    inputs, outputs = FilesReader.read_dataset_vectorization(arquivo)

    exemplo_back_um(layers,thetas)

    lamb, layers = FilesReader.read_networks("network2.txt")
    thetas = FilesReader.read_thetas("initial_weights2.txt")
    #exemplo_back_two(layers, thetas)