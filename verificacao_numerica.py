

def verificacao_numerica():






if __name__ == '__main__':
    arquivo = "pima.tsv"
    lamb, layers = FilesReader.read_networks("network.txt")
    data, attribute_matrix = FilesReader.read_dataset(arquivo)
    print(lamb)
    print(layers)
