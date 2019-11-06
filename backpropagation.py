import NeuralNetwork as NN
import FilesReader

def backpropagation(batch,num_layers):
    pass

if __name__ == '__main__':
    arquivo = "pima.tsv"
    lamb, layers = FilesReader.read_networks("network.txt")
    data, attribute_matrix = FilesReader.read_dataset(arquivo)
    print(lamb)
    print(layers)
