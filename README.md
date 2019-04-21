# Smile Detector using CNN with Keras 

## Execução (Run)


Primeiro instale as bibliotecas necessárias executando o comando abaixo (recomendo utilizar o python3.6 que foi a versão usada no desenvolvimento):

> pip install -r req.txt


Pronto, agora basta executar o arquivo app.py:


> python app.py

## Overview 

Foi utilizado a linguagem python para o desenvolvimento do programa e o banco de dados <banco>, algumas imagens do banco de dados:
  

Primeiramente foi relizado a leitura dos documentos .txt que indicava quais imagens deveriam ser usadas no teste. Assim foi gerados 2 listas, uma com nomes das imagens com pessoas sorrindo e outra que não, em seguida foi dividido cada lista em dados de treinamento, teste e validação. Obtendo assim 6 listas:

- train_smile 
- test_smile 
- val_smile

- train_nosmile  
- test_nosmile 
- val_nosmile

O próximo passo foi criar as pastas que receberá as imagens de cada lista, então é criado os diretorios Train, Test e Val e dentro de cada um foi criado mais duas pasta smile e nosmile. Em seguida, é feito uma copia da imagens do dataset original <data> e é movido essas copias para as pastas de acordo com a lista que ela pertence.
  
É realizado esse procedimento, para que os dados estejam de acordo com o que a função de pré processamento do pacote Keras ImageDataGenerator.flow_from_directory solicita, foi usado essa método pois ela ajuda no pré processamento da imagens. Além disso, devido o nosso conjunto de dados não ser muito grande, realizamos com essa função o aumento artificialmente do conjunto de dados. 


O para realizar detecção para saber se as imagens são de pessoas sorrindo ou não, foi utilizando modelo de Rede Neural Convolucional (CNN), resumo da rede:

<image>

Foi colocado para relizar o treinamento durante 10 epocas, porém devido para um parametro adicionado EarlyStopping, que ele para o treinamento se caso começar a dectar overfitting.

<image trainamento>
  
<grafico>
  
Após o treinamento foi feito a predição no dados de teste no qual conseguiu um precisão de 96,04% de acerto na classificação.

## Especificações da máquina utilizada para o treinamento  

-OS: Ubuntu 16.04 x64
-RAM: 8G
-Processador; Intek Core i5 2.5GHz x4
-PU: Nvidia GeForce 940MX 2G

O código foi executado tanto na GPU Nvidia quanto na CPU Intel.

Tempo de treimanto usando CPU: Cerca de 4 minutos e 22 segundos por época

Tempo de treimanto usando GPU: Cerca de 48 segundos por época
