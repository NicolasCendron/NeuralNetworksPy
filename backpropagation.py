import FilesReader
import numpy as np
import math
import NeuralNetwork
import sys

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_vetor = np.vectorize(sigmoid)


def propagation(exemplo, thetas, network):
    entrada = list(exemplo[0])

    ativacao = []
    ativacao.append(np.array([1] + entrada))
    Z = []
    for i in range(1,len(network) - 1):
        Zatual = (thetas[i-1]).dot(ativacao[i-1])
        Z.append(Zatual)
        ativacaoAtual = np.insert(sigmoid_vetor(Zatual), 0, 1)
        ativacao.append(ativacaoAtual)

    ZFinal = thetas[-1].dot(ativacao[-1])
    ativacao_final = sigmoid_vetor(ZFinal)

    print(ativacao)
    return ativacao, ativacao_final

def calculaJ(mini_batch, thetas, regularization, network):
    J = 0
    cont = 0
    for example in mini_batch:
        cont += 1

        inputs = np.array(example[0])
        outputs = np.array(example[1])

        ativacao,predicted_output = propagation(example, thetas, network)

        vectorJ =  np.multiply(np.negative(outputs),np.log(predicted_output))
        vectorJ -= np.multiply((np.ones(outputs.size) - outputs),np.log(np.ones(predicted_output.size) - predicted_output))

        J += np.sum(vectorJ)

    J = J/ len(mini_batch)
    S = 0
    for theta_matrix in thetas:
        for theta_line in theta_matrix:
            for i in range(1,len(theta_line)): #Evita thetas de bias
                S+= math.pow(theta_line[i],2)
    S = regularization/(2*len(inputs))*S
    return J + S


def backpropagation(exemplos, thetas, regularizacao, network, learning_rate):
    J = 0
    cont = 0

    D = []
    for i in range(len(network) - 1):
        D.append([])

    for exemplo in exemplos:
        cont += 1
        print("Calculando gradientes com base no exemplo  " + str(cont))
        saidas = np.array(exemplo[1])

        # 1.1 Propaga x(i) e obtém as saídas f(x(i)) preditas pela rede
        ativacao, saidas_preditas = propagation(exemplo, thetas, network)

        # 1.2 calcula deltas para os neurônios da camada de saída
        error = saidas_preditas - saidas

        #Error tá correto

        # cria array para armazenar os deltas
        deltas = []
        for i in range(len(network)):
            deltas.append([])
        deltas[-1] = error
        print("delta"+str(len(network)))
        print(error)
        # 1.3 Para cada camada k=L-1…2, calcula os deltas para as camadas ocultas

        for k in reversed(range(1, len(network)-1)):

            delta = np.transpose(thetas[k]).dot(deltas[k+1])  #.* ativacao[k] .* (1-ativacao[k])
            delta = np.multiply(delta, ativacao[k])
            delta = np.multiply(delta, (1 - ativacao[k]))

            #; Remove o primeiro elemento de delta(l=k) (i.e., o delta associado ao neurônio de bias da camada k
            deltas[k] = delta[1:]
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
            print("Gradientes de Theta" + str(k + 1) + " com base no exemplo" + str(cont))
            print(factor)

            #D[k] = factor
            if len(D[k]) == 0:
                D[k] = factor
            else:
               D[k] += factor

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
    print("Dataset completo processado. Calculando gradientes regularizados")
    for k in range(0, len(network)-1):

        gradiente = np.multiply(learning_rate, D[k])
        print("Gradientes finais para Theta" + str(k+1) + " (com regularizacao):")
        print(gradiente)
        thetas[k] = thetas[k] - gradiente

    return thetas


def exemplo_back(layers,lamb, theta_matrices,instancias):
    # 3 camadas [1 2 1]

    network = np.array(layers)

    theta1 = theta_matrices[0]
    #theta1 = np.array([ [0.4 , 0.1] ,
    #                    [0.3 , 0.2]])

    theta2 = theta_matrices[1]
    #theta2 = np.array([0.7, 0.5, 0.6])

    thetas = np.array([theta1,theta2])

    regularizacao = lamb
    learning_rate = 1

    #entradas = [[0.13], [0.42]]
    #saidas = [[0.9], [0.23]]
    exemplos = instancias
    print(exemplos)
    #exemplos = []
    #for i in range(0,len(instancias)):
     #   exemplos.append([instancias[0][i], instancias[1][i]])

    novos_thetas = backpropagation(exemplos, thetas, regularizacao, network, learning_rate)
    return novos_thetas

def escreve_novos_thetas(thetas):
    print(thetas)
    nome_arquivo = "resultado_backpropagation.txt"
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
        #python3 backpropagation.py network_teste1.txt initial_weights_teste1.txt dataset_teste1
        #python3 backpropagation.py network_teste1.txt resultado_backpropagation.txt dataset_teste1
        network_file = sys.argv[1]
        weights_file = sys.argv[2]
        dataset_file = sys.argv[3]

        arquivo = "pima.tsv"
        lamb, layers = FilesReader.read_networks(network_file)
        thetas = FilesReader.read_thetas(weights_file)
        instancias = FilesReader.read_simple_dataset(dataset_file)
        novos_thetas = exemplo_back(layers, lamb, thetas,instancias)

        escreve_novos_thetas(novos_thetas)

        #lamb, layers = FilesReader.read_networks("network.txt")
        #thetas = FilesReader.read_thetas("initial_weights.txt")
        #exemplo_back_um(layers,lamb,thetas)
        #NeuralNetwork.neural_network(layers, lamb, thetas, [[0.13], [0.42]], [[0.9], [0.23]])

