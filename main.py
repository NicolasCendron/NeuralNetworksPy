import FilesReader

if __name__ == '__main__':
    arquivo = "pima.tsv"
    #nos arquivos abaixo os cabeçalhos estão na primeira linha para facilitar a nomeação das colunas, avisar no relatório!
    #arquivo = "wine.data"
    #arquivo = "ionosphere.data"
    lamb, layers = FilesReader.read_networks("network.txt")
    data, attribute_matrix = FilesReader.read_dataset(arquivo)
