import Util as ut
import NeuralNetwork as nn
import backpropagation as bp
import copy
import numpy as np
def generate_partitions(data, K):
    partitions = []
    ordered_data = data
    # data sorted by output, for use with data with single output only!
    ordered_data.sort(key = lambda ordered_data: ordered_data[1])

    # generate partitions
    for p in range(K):
        partitions.append([])

    # populate partitions
    for i in range(len(ordered_data)):
        partitions[i % K].append(ordered_data[i])

    return partitions

def run(data,thetas, regularization, network,dataset_file):
    K = 10

    if(dataset_file == "wine.data"): #ok
        min_diff = 0.001
        B = 10

    if(dataset_file == "wdbc.data"): #ok
        B = 12
        min_diff = 0.001

    if(dataset_file == "pima.tsv"): #mais do que 0.15 de regularizacao piora muito os resultados
        #melhores valores que encontrei, mas a performance ainda está abaixo dos outros datasets
        B = 5
        min_diff = 0.00002

    if(dataset_file == "ionosphere.data"): #ok melhor B = 2 ou 5
        B = 5
        min_diff = 0.0002

    partitions = generate_partitions(data, K)
    learning_rate = 0.5

    for i in range(K):
        print("Running K = " + str(i))

        novos_thetas = copy.deepcopy(thetas)
        evaluation = partitions[i]
        training = []
        for p in range(K):
            if p != i:
                training = training + partitions[p]
        batch_p = generate_partitions(training, B)

        # using b mini-batches
        for p in range(B):
            cont = 1
            initial_j_value = nn.calculate_j(batch_p[p], novos_thetas, regularization, network)
            #print("j inicial:" + str(initial_j_value))
            while (cont < 1000):
                novos_thetas, gradientes = bp.backpropagation(batch_p[p], novos_thetas, regularization, network,
                                                          learning_rate, 0)
                cont += 1
                if (cont % 20) == 0: #60
                    j_value = nn.calculate_j(batch_p[p], novos_thetas, regularization, network)
                    #print("j value:" + str(j_value))
                    if(initial_j_value >= j_value):
                        diff = initial_j_value - j_value
                    else:
                        diff = j_value - initial_j_value
                    #print("diferenca: " + str(diff))
                    #print("performance batch:" + str(p))
                    #performance = calculate_performance(network, evaluation, novos_thetas)
                    if (diff < min_diff):
                        #print("performance batch: " + str(p))
                        #performance = calculate_performance(network, evaluation, novos_thetas)
                        break
                    initial_j_value = j_value
                #if(cont == 999):
                    #print("estouro batch: " + str(p))
                    #performance = calculate_performance(network, evaluation, novos_thetas)


        # generate cross validation training (K-1) and evaluation (1) partitions
        '''
        training = []
        for p in range(K):
            if p != i:
                training = training + partitions[p]
        
        cont = 1
        max_iteracoes = 1000
        while (cont < max_iteracoes):
            novos_thetas, gradientes = bp.backpropagation(training, novos_thetas, regularization, network,
                                                          learning_rate, 0)
            if (cont % 10) == 0:
                performance = calculate_performance(network, evaluation, novos_thetas)
                if(performance[2] > 0.95):
                    break

            cont += 1'''

        respostas = []

        for i in range(len(evaluation)):

            if network[-1] == 1: # Se possui apenas uma saida
                resposta_certa = evaluation[i][1][0]
                resposta_rede = nn.evaluate(evaluation[i],novos_thetas,network)
                resposta_rede_int = round(resposta_rede[0])
                respostas.append( (resposta_certa, resposta_rede_int) )

            else: # Se possui mais de uma saída
                index_resposta_certa = evaluation[i][1].index(max(evaluation[i][1]))

                respostas_rede = nn.evaluate(evaluation[i],novos_thetas,network)
                index_resposta_rede = np.where( respostas_rede == max(respostas_rede))[0][0]
                respostas.append( (index_resposta_certa,index_resposta_rede) )

        print(respostas)
        if network[-1] == 1: # Se possui apenas uma saida
            resultado = ut.performance_binary(respostas,[0,1])
        else:  # Se possui mais de uma saída
            resultado = ut.performance_multiclass(respostas,[0,1,2])

        print("Precision , Recall , F1-Score")
        print(resultado)
        #generate batches instead of using training directly
        #j_value = nn.j_function(training, thetas, regularization, network)

    return novos_thetas, gradientes

def calculate_performance(network,evaluation,novos_thetas):
    respostas = []
    resultado = []

    for i in range(len(evaluation)):

        if network[-1] == 1:  # Se possui apenas uma saida
            resposta_certa = evaluation[i][1][0]
            resposta_rede = nn.evaluate(evaluation[i], novos_thetas, network)
            resposta_rede_int = round(resposta_rede[0])
            respostas.append((resposta_certa, resposta_rede_int))

        else:  # Se possui mais de uma saída
            index_resposta_certa = evaluation[i][1].index(max(evaluation[i][1]))

            respostas_rede = nn.evaluate(evaluation[i], novos_thetas, network)
            index_resposta_rede = np.where(respostas_rede == max(respostas_rede))[0][0]
            respostas.append((index_resposta_certa, index_resposta_rede))

    correct = 0
    for r in respostas:
        if(r[0] == r[1]):
            correct += 1

    print("respostas corretas do total: " + str(correct) + "/" + str(len(respostas)))
    #print(respostas)
    if network[-1] == 1:  # Se possui apenas uma saida
        resultado = ut.performance_binary(respostas, [0, 1])
    else:  # Se possui mais de uma saída
        resultado = ut.performance_multiclass(respostas, [0, 1, 2])

    return resultado