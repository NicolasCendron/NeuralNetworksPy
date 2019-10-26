import CsvReader

if __name__ == '__main__':
    arquivo = "pima.tsv"
    data, attribute_matrix = CsvReader.read_csv(arquivo)
