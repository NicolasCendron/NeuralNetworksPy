import numpy as np
import math
import CrossValidation as cv

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


def j_function_old(mini_batch, thetas, regularization, network):
    J = 0
    cont = 0
    for example in mini_batch:
        cont += 1
        print("Processando exemplo de treinamento " + str(cont))

        inputs = np.array(example[0])
        outputs = np.array(example[1])

        predicted_output = propagation(example, thetas, network)

        print("Saidas Preditas para o Exemplo" + str(cont))
        print(predicted_output)
        print("Saidas Esperadas para o Exemplo" + str(cont))
        print(outputs)

        vectorJ = (np.negative(outputs)).dot(np.log(predicted_output))
        vectorJ -= (np.ones(outputs.size) - outputs).dot(np.log(np.ones(predicted_output.size) - predicted_output))

        print("J para o Exemplo" + str(cont))
        print(np.sum(vectorJ))

        J += np.sum(vectorJ)

    J = J/ len(mini_batch)
    S = 0
    for theta_matrix in thetas:
        for theta_line in theta_matrix:
            for i in range(1,len(theta_line)): #Evita thetas de bias
                S+= math.pow(theta_line[i],2)
    S = regularization/(2*len(inputs))*S

    print("J para o total do dataset")
    print(str(J + S))
    return J + S

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

    return ativacao, ativacao_final


def propagation_old(example,thetas,network):
    input = list(example[0])
    print("propagando entrada" + str(example))

    activation = []
    activation.append([1] + input)
    print("ativacao 1")
    print(activation[0])
    z = []
    for i in range(1,len(network) - 1):

        z_current = (thetas[i-1]).dot(activation[i-1])

        print("z" + str(i + 1))
        print(z_current)

        z.append(z_current)
        activation_current = np.insert(sigmoid_vetor(z_current), 0, 1)
        activation.append(activation_current)

        print("a" + str(i + 1))
        print(activation[-1])

    z_final = thetas[-1].dot(activation[-1])
    activation_final = sigmoid_vetor(z_final)

    print("ZFinal")
    print(z_final)
    print("AFinal")
    print(activation_final)

    return activation_final

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
    #cv.run(examples,thetas, regularization, network)
    j_value = j_function(examples, thetas, regularization, network)

if __name__ == '__main__':
    pass