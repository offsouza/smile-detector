# Smile Detector using CNN with Keras 

## Execução (Run)


Primeiro instale as bibliotecas necessárias executando o comando abaixo (recomendo utilizar o python3.6 que foi a versão usada no desenvolvimento):

> pip install -r req.txt

Depois, extraia o arquivo que contém as imagens `lfwcrop_grey.zip`.

Pronto, agora basta executar o arquivo app.py,


> python app.py

ou executar o jupyter notebook `App.ipynb`.

## Overview 

Foi utilizado a linguagem python para o desenvolvimento do programa e o banco de dados [LFWcrop Face Dataset](http://conradsanderson.id.au/lfwcrop/). Abaixo algumas imagens do banco de dados:
  
<p align="center">
  <img width="360" height="288" src="https://github.com/offsouza/smile-detector/blob/master/images/faces.png">
</p>
  

Primeiramente foi relizado a leitura dos documentos .txt que indicava quais imagens deveriam ser usadas no teste. Assim foi gerados 2 listas, uma com nomes das imagens com pessoas sorrindo e outra que não, em seguida foi dividido cada lista em dados de treinamento, teste e validação. Obtendo assim 6 listas:

- train_smile 
- test_smile 
- val_smile

- train_nosmile  
- test_nosmile 
- val_nosmile

O próximo passo foi criar as pastas que receberá as imagens de cada lista, então é criado os diretorios Train, Test e Val e dentro de cada uma foi criado mais duas pasta smile e nosmile. Em seguida, é feito uma copia da imagens do dataset original e é movido essas copias para as pastas de acordo com a lista que ela pertence.
  
É realizado esse procedimento para que os dados estejam de acordo com o que a função de pré processamento ` ImageDataGenerator.flow_from_directory` do pacote `Keras` solicita, foi usado esse método pois ele auxilia no pré processamento da imagens. Além disso, devido o nosso conjunto de dados não ser muito grande, realizamos também com essa função o aumento artificialmente do conjunto de dados. 

Para realizar a predição a fim de saber se as imagens são de pessoas sorrindo ou não, foi utilizando modelo de Rede Neural Convolucional (CNN) em que podemos ver resumo da rede abaixo:

<p align="center">
  <img width="548" height="472" src="https://github.com/offsouza/smile-detector/blob/master/images/summary">
</p>

Foi colocado para relizar o treinamento durante 10 epocas, porém devido para um parâmetro adicionado `EarlyStopping` em que ele para o treinamento caso a métrica `loss` esteja aumentando ao invés de estar diminuindo, que pode ser um sinal de overfitting, assim o treinamento foi parado na quinta época de treinamento.

<p align="center">
  <img width="510" height="250" src="https://github.com/offsouza/smile-detector/blob/master/images/fit">
</p>
  
<p align="center">
  <img width="800" height="500" src="https://github.com/offsouza/smile-detector/blob/master/images/plot1.png">
</p>
  
Após o treinamento, o modelo obteve precisão na classificação de 98,91% nos dados de treinamento, 96,46% nos dados de validação e 96,04% nos dados de teste.

<p align="center">
  <img width="306" height="275" src="https://github.com/offsouza/smile-detector/blob/master/images/smile.png">
</p>

## Especificações da máquina utilizada para o treinamento  

- OS: Ubuntu 16.04 x64
- RAM: 8Gb
- Processador: Intel Core i5 2.5GHz x4
- GPU: Nvidia GeForce 940MX 2G

O código foi executado tanto na GPU Nvidia quanto na CPU Intel.

Tempo de treinamento usando CPU: Cerca de 4 minutos e 22 segundos por época

Tempo de treinamento usando GPU: Cerca de 48 segundos por época
