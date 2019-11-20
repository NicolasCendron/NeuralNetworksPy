import numpy as np
import math
import CrossValidation as cv
import copy
import sys
import FilesReader
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_vetor = np.vectorize(sigmoid)

def evaluate_mini_batch():
    pass

def numeric_gradient():
    pass

def feature_normalization():
    pass

def create_batches():
    pass

def evaluate(exemplo, thetas, network):
    ativacao, ativacao_final = propagation(exemplo, thetas, network)
    return ativacao_final

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

def propagation(exemplo, thetas, network):
    entrada = list(exemplo[0])

    ativacao = []
    ativacao.append(np.array([1] + entrada))
    Z = []
    for i in range(1,len(network) - 1):
        #print(thetas[i - 1])
        #print(ativacao[i-1])
        Zatual = (thetas[i-1]).dot(ativacao[i-1])
        Z.append(Zatual)
        ativacaoAtual = np.insert(sigmoid_vetor(Zatual), 0, 1)
        ativacao.append(ativacaoAtual)

    ZFinal = thetas[-1].dot(ativacao[-1])
    ativacao_final = sigmoid_vetor(ZFinal)

    return ativacao, ativacao_final



def neural_network(layers,lamb, theta_matrices,inputs, outputs):
    network = np.array(layers)

    thetas = [theta_matrices[0]]
    cont = 0

    for theta in theta_matrices:
        if cont == 0:
            cont += 1
            continue

        thetas.append(theta)

    thetas = np.array(thetas)

    regularization = lamb

    examples = []

    for i in range(0, len(inputs)):
        examples.append([inputs[i], outputs[i]])

    #sort examples
    #get training set from examples - cross validation
    thetas_finais, gradientes_finais = cv.run(examples,thetas, regularization, network)
    j_value = calculaJ(examples, thetas_finais, regularization, network)
    print(j_value)
    return thetas_finais, gradientes_finais



