# -*- coding: utf-8 -*-

# Código desenvolvido para converter uma database de imagens em códigos de incorporações
# que depois podem ser comparados a fim de validar se um rosto pertence a uma pessoa 
# ou não.

# @Autor: rodrigo.ferrari
# @Editor: Thiago Luis Pinho
# Last edit: 2019/10/29



import keras_vggface
# Importa a rede neural já treinada que gera os códigos de incorporação.
# Caso você queira saber a versão, basta descomentar o código abaixo.
# print(keras_vggface.__version__),

import argparse
import pickle
from encoding_helpers import get_embeddings, store_codes_with_names

# Parte do código responsável por criar e configurar os argumentos para a chamada no terminal
ap = argparse.ArgumentParser()
ap.add_argument("-db", "--database", default="database_labeled",
  help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", default="encodings.pickle",
  help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
  help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# Converte as imagens da database em códigos de incorporação
embbedings, person_names, filenames = get_embeddings(args["database"])


# Itera sobre os códigos de incorporações

data = store_codes_with_names(embbedings, person_names, filenames)

# Armazena os nomes e códigos gerados
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()