# NeuralNetworksPy

1  ObjetivoO Trabalho 2 da disciplina consiste na implementação de uma rede neural treinada viabackpropa-gation. O trabalho será desenvolvido emgrupos de até 3 alunose, conforme discutido em aula,inclui a implementação de todas as características principais de redes neurais discutidas ao longo dosemestre. Em particular, os grupos deverão implementar:

•uma rede neural composta por um número ajustável de neurônios e camadas e treinada viabackpropagation;
•funcionalidade que permita, via linha de comando, informar a sua implementação a estru-tura de uma rede de teste (i.e., estrutura de camadas/neurônios, pesos iniciais, e fator deregularização), e um conjunto de treinamento, e que retorne o gradiente calculado para cadapeso;
•funcionalidade que permita, via linha de comando, efetuar averificação numéricado gradiente,a fim de checar a corretude da implementação de cada grupo;
•um mecanismo para normalização das features/dados de treinamentos;
•mecanismo para uso de regularização.


•Resumo dos itens/requisitos descritos acima, e que serão explicitamente avaliados:
1.(obrigatório)Verificação numérica do gradiente;
2.(obrigatório)Interface de linha de comando para teste dobackpropagation;
3.(obrigatório)Backpropagationimplementado de forma completa e correta;
4.(obrigatório)Suporte a regularização;
5.(obrigatório)k-folds+F1 score;
6.(obrigatório)Testes (para cada dataset) com diferentes arquiteturas de rede;
7.(obrigatório)Testes (para cada dataset) com diferentes meta-parâmetros:  taxa deaprendizado, taxa de regularização, etc;
8.(obrigatório)Curvas de aprendizado mostrando a função de custoJpara cada dataset;
9.(obrigatório)Análise dos datasets 1-3;
10.(obrigatório)Pontualidade na entrega do trabalho.  Atrasos na entrega do trabalhoserão penalizados proporcionalmente ao tempo de atraso, sendo descontado 1 (um) pontopor dia de atraso (o trabalho como um todo vale 10 pontos);
11.(obrigatório)Apresentação oral dos resultados: qualidade da apresentação e domínioda implementação e resultados, bem como capacidade de arguição acerca dos mesmos;
12.(obrigatório)Qualidade do relatório final;
13.(opcional) Análise do dataset 4;
14.(opcional) Suporte a treinamento mini-batch;
15.(opcional) Suporte a vetorização;
16.(opcional) Suporte ao método do momento.
