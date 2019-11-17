import Util as ut
import NeuralNetwork as nn
import backpropagation as bp
import copy

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
        max = 100
        novos_thetas = copy.deepcopy(thetas)
        while(cont < max):
            novos_thetas, gradientes = bp.backpropagation(training,novos_thetas,regularization,network,learning_rate,0)
            cont+=1

        respostas = []

        #fiz pensando so no wine
        '''
        for i in range(len(training)):

            if len(training[i][1]) == 1:
                resposta_certa = training[i][1][0]
                resposta_rede = round_of_rating(nn.evaluate(training[i],novos_thetas,network)[0])
                print("Respostas")
                print(resposta_certa)
                print(resposta_rede)
                respostas.append([resposta_certa, resposta_rede])

            else:
                resposta_certa = training[i][1].index()

                respostas_rede = nn.evaluate(exemplo,novos_thetas,network)
                index_resposta_rede = respostas_rede.index(max(respostas_rede))

            respostas.append( (index_resposta_certa,index_resposta_rede) )

        ut.performance_multiclass(respostas,[1,2,3])
        '''

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
