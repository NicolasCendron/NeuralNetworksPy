import csv
from collections import OrderedDict
import numpy as np

def read_dataset(arquivo):
    with open(arquivo) as dataFile:
        validation_data = []
        if arquivo == "pima.tsv":
            csvReader = csv.reader(dataFile, delimiter='\t')
        else:
            csvReader = csv.reader(dataFile, delimiter=',')
        cont = 0
        attribute_matrix = []
        for row in csvReader:
            if cont == 0:
                for i in range(len(row)):
                    if row[i] == "target":
                        attribute_matrix.append(["Class", []])
                    else:
                        attribute_matrix.append([row[i], []])
                cont+=1
                continue
            dado_atual = OrderedDict()

            for i in range(len(row)):
                dado_atual[attribute_matrix[i][0]] = row[i]
                if row[i] not in attribute_matrix[i][1]:
                    attribute_matrix[i][1].append(row[i])

            validation_data.append(dado_atual)

        target_attribute_index = len(attribute_matrix) - 1

        if arquivo == "ionosphere.data":
            attribute_matrix[-1][1] = ["g", "b"]
        elif arquivo == "wine.data":
            attribute_matrix[0][1] = ["1", "2", "3"]
            target_attribute_index = 0
        else:
            attribute_matrix[-1][1] = ["1", "0"]

        '''
        for attribute in attribute_matrix:
            if attribute_matrix.index(attribute) != target_attribute_index:
                try:
                    x = float(attribute[1][0])
                    sum = 0
                    string1 = attribute[0]
                    for v in validation_data:
                        sum += float(v[string1])

                    length = len(validation_data)
                    average = sum / length
                    attribute[1] = ["@< " + str(round(average,3)), "@> " + str(round(average,3))]
                except ValueError:
                    continue
        '''
        return validation_data, attribute_matrix

def read_networks(arquivo):
    with open(arquivo) as f:
        lamb = float(f.readline())
        layers = []
        for line in f:
            layers.append(int(line))

    return lamb, layers

def read_thetas(arquivo):
    with open(arquivo) as f:

        matrix_list = []
        for line in f:
            line = line.strip()
            theta = line.split(";")
            theta_matrix = []
            for row in theta:
                values = row.split(",")
                matrix = []
                for value in values:
                    matrix.append(float(value))
                theta_matrix.append(matrix)
            matrix_list.append(np.array(theta_matrix))

        return matrix_list

def read_dataset_vectorization(arquivo):
    with open(arquivo) as dataFile:
        inputs = []
        outputs = []
        if arquivo == "pima.tsv":
            csvReader = csv.reader(dataFile, delimiter='\t')
        else:
            csvReader = csv.reader(dataFile, delimiter=',')
        cont = 0

        for row in csvReader:

            if cont == 0:
                cont+=1
                continue

            if arquivo == "wine.data":
                outputs.append(row[0])
                inputs.append(row[1:])
            else:
                inputs.append(row[:-1])
                outputs.append(row[-1])

        if arquivo == "ionosphere.data":
            #set outputs to g = 0 and b = 1
            float_outputs = []

            for output in outputs:
                if output == "g":
                    float_outputs.append("0")
                else:
                    float_outputs.append("1")

            return inputs, float_outputs

        else:
            return inputs, outputs
