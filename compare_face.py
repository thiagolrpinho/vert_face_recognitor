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

# Parte do código responsável por criar e configurar os argumentos para a chamada no terminal
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="database_test",
  help="path to input database of unkown images")
ap.add_argument("-e", "--encodings", default="encodings.pickle",
  help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
  help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

unknown_encodings, unknown_names, unknown_filenames = get_embeddings(args["image"])

# Organizamos as prováveis pessoas por nome com seus respectivos códigos e nome do arquivo
data_unknown = store_codes_with_names(unknown_encodings, unknown_names, unknown_filenames)
print(data_unknown)

for unknown_person in data_unknown.keys():
  # Primeiro procuramos em database pela pessoa a ser  comparada
  person_encodings = data[ unknown_person ]
  
  for encoding, filename in data_unknown[unknown_person]:
  # Depois iteramos pelas imagens a serem comparadas da mesma pessoa
    matches = []
    for known_encoding, known_filename in person_encodings:
      # Nós armazenamos os resultados em uma lista
      matches.append( is_match( encoding, known_encoding ) )

    if( matches.count(True) > len(matches)/2 ):
      # Se mais que a metade dos rostos baterem, consideramos a pessoa como a mesma
      print( "A imagem " + filename + " é do " + unknown_person )
    else: 
      print( "A imagem " + filename + " não é do " + unknown_person )

