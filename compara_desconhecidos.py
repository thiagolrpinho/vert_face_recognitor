# -*- coding: utf-8 -*-

# Código desenvolvido para comparar uma imagem de um rosto( o nome do arquivo deve ser o nome da pessoa)
# com uma database já convertida em código de incorporação
# e valida se é a mesma pessoa ou não.

# @Autor: rodrigo.ferrari
# @Editor: Thiago Luis Pinho
# Last edit: 2019/10/29


import argparse
import pickle
from encoding_helpers import get_embeddings, is_match, store_codes_with_names
from PIL import Image, ExifTags
import datetime
import os
from imutils import paths

# Parte do código responsável por criar e configurar os argumentos para a chamada no terminal
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="database_comparacao",
  help="path to input database of unkown images")
ap.add_argument("-e", "--encodings", default="encodings.pickle",
  help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
  help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] Carregando dados de pessoas conhecidas...")
dados_conhecidos = pickle.loads(open(args["encodings"], "rb").read())

unknown_encodings, unknown_names, unknown_filenames, unknown_images = get_embeddings(args["image"])

# Organizamos as prováveis pessoas por nome com seus respectivos códigos e nome do arquivo
dados_comparacao = store_codes_with_names(unknown_encodings, unknown_names, unknown_filenames, unknown_images)

for pessoa_comparada in dados_comparacao.keys():
  # Primeiro procuramos em database pela pessoa a ser  comparada
  if pessoa_comparada in dados_conhecidos:
    dados_por_pessoa = dados_conhecidos[ pessoa_comparada ]
    
    for codigo_comparado, nome_arquivo_comparado, imagem_comparada in dados_comparacao [pessoa_comparada]:
    # Depois iteramos pelas imagens a serem comparadas da mesma pessoa
      matches = []
      for codigo_conhecido, nome_arquivo_conhecido in dados_por_pessoa:
        # Nós armazenamos os resultados em uma lista
        matches.append( is_match( codigo_comparado, codigo_conhecido ) )

      if( matches.count(True) >= len(matches)/2 ):
        # Se mais que a metade dos rostos baterem, consideramos a pessoa como a mesma
        print( "A imagem " + nome_arquivo_comparado + " é do " + pessoa_comparada )
      else: 
        print( "A imagem " + nome_arquivo_comparado + " não é do " + pessoa_comparada )
        now = datetime.datetime.now()
        imagem_comparada.save("./database_vetadas/" + str(now.day) + "_" + str(now.hour) + "h" + str(now.minute) + 'm' + str(now.second) + "s_"  + nome_arquivo_comparado[:-5] + "_vetado" + ".jpg")
        imagePath = list(paths.list_images( "./database_conhecidos/" + nome_arquivo_conhecido[:-5] ))[0]
        imagem_conhecida = Image.open(imagePath)
        imagem_conhecida.save("./database_vetadas/" + str(now.day) + "_" + str(now.hour) + "h" + str(now.minute) + 'm' + str(now.second) + "s_"  + nome_arquivo_comparado[:-5] + "_original" + ".jpg")


  else: 
    # Caso a pessoa não seja encontrada na base de dados conhecida, ela não é comparada 
    # e um aviso é emitido no terminal
    dados_pessoa_sem_comparacao = dados_comparacao[pessoa_comparada]
    print("A pessoa " + pessoa_comparada + " não foi encontrada na base de dados conhecida.")
    for codigo_sem_comparacao, nome_arquivo_nao_comparado, imagem_sem_comparacao in dados_pessoa_sem_comparacao:
      print("A imagem com nome " + nome_arquivo_nao_comparado + " não teve uma pessoa encontrada na database correspondente")

