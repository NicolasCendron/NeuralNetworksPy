import CsvReader

if __name__ == '__main__':
    arquivo = "pima.tsv"
    #nos arquivos abaixo os cabeçalhos estão na primeira linha para facilitar a nomeação das colunas, avisar no relatório!
    #arquivo = "wine.data"
    #arquivo = "ionosphere.data"
    data, attribute_matrix = CsvReader.read_csv(arquivo)
