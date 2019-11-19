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

def run(data,thetas, regularization, network):
    fixedSeed = 0
    seed = 7
    K = 10
    stats = []

    partitions = generate_partitions(data, K)
    learning_rate = 1;

    for i in range(K):
        print("Running K = " + str(i))

        # generate cross validation training (K-1) and evaluation (1) partitions
        training = []
        for p in range(K):
            if p != i:
                training = training + partitions[p]
        evaluation = partitions[i]

        cont = 1;
        max_iteracoes = 1000
        novos_thetas = copy.deepcopy(thetas)
        while(cont < max_iteracoes):
            novos_thetas, gradientes = bp.backpropagation(training,novos_thetas,regularization,network,learning_rate,0)
            cont+=1

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

        print(resultado)
        #generate batches instead of using training directly
        #j_value = nn.j_function(training, thetas, regularization, network)
        '''
        # run bootstrap and create each decision tree of forest
        for t in range(ntree):
            training_set = bs.generate_training_set(training, fixedSeed)
            #test_set = bs.generate_test_set(training, training_set)
            fixedSeed += len(data)

            decisionTree = dt.DecisionTree()

            # m attributes are used
            #reduced_matrix = vd.select_m_attributes(attribute_matrix), this is being done on the select_node_id function
            seed += len(data)

            dt.select_node_id(decisionTree, training_set, attribute_matrix,False)
            dt.add_branch(decisionTree, training_set,attribute_matrix)
            dt.split_examples(decisionTree, training_set, attribute_matrix,False)


            #print("root attribute selected:" + decisionTree.node_id)

            #dt.print_tree(decisionTree)
            forest.append(decisionTree)

            all_classes = ut.get_classes(attribute_matrix)
            #ut.evaluate_tree(decisionTree, test_set, all_classes)

        stats.append(ut.evaluate_forest(forest, evaluation, all_classes))
    ut.print_stats(stats, all_classes)
    '''

    return novos_thetas, gradientes
