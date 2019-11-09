import FilesReader

def verificacao_numerica():
    pass





if __name__ == '__main__':
    arquivo = "pima.tsv"
    lamb, layers = FilesReader.read_networks("network.txt")
    inputs, outputs = FilesReader.read_dataset_novo(arquivo)
    print(inputs)
    #print(outputs)