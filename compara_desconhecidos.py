# # Compara Desconhecidos
#  -*- coding: utf-8 -*-
# 
# Código desenvolvido para comparar uma imagem de um rosto( o nome do arquivo deve ser o nome da pessoa)
# com uma database já convertida em código de incorporação
# e valida se é a mesma pessoa ou não.
#  
# 
# **Autor:** Rodrigo Ferrari 
# 
# **Editores**: Thiago Luis Pinho, Lucas Moutinho
# 
# **Last edit:** 2019/12/02

import argparse
import pickle
from encoding_helpers import get_embeddings, is_match, store_codes_with_names
from image_helpers import save_image


# ## Configurando variáveis de comportamento para o código


args = {"image" : "database_comparacao" # Caminho para a database de imagens a serem comparadas"
        , "encodings" : "encodings.pickle" # Caminho para uma database serializada com os códigos de incorporação das faces
        , "detection-method" : "cnn"    # Modelo de detecção facial a ser usado 'cnn' ou 'hog'
}


# ### Carregando códigos de Incorporação da base de dados conhecida


print("[INFO] Carregando dados de pessoas conhecidas...")
dados_conhecidos = pickle.loads(open(args["encodings"], "rb").read())
unknown_encodings, unknown_names, unknown_filenames, unknown_images = get_embeddings(args["image"])


# ### Comparamos as pessoas com seus respectivos registros na base de dados conhecida


# Organizamos as prováveis pessoas por nome com seus respectivos códigos e nome do arquivo
dados_comparacao = store_codes_with_names(unknown_encodings, unknown_names, unknown_filenames, unknown_images)

for pessoa_comparada in dados_comparacao.keys():
    # Primeiro procuramos em database pela pessoa a ser comparada
    if pessoa_comparada in dados_conhecidos:
        dados_por_pessoa = dados_conhecidos[ pessoa_comparada ]
        
        for codigo_comparado, nome_arquivo_comparado, imagem_comparada in dados_comparacao [pessoa_comparada]:
            # Depois iteramos pelas imagens a serem comparadas da mesma pessoa
            matches = []
            for codigo_conhecido, nome_arquivo_conhecido, imagem_conhecida in dados_por_pessoa:
                # Nós armazenamos os resultados em uma lista
                matches.append( is_match( codigo_comparado, codigo_conhecido ) )

            if( matches.count(True) >= len(matches)/2 ):
                # Se mais que a metade dos rostos baterem, consideramos a pessoa como a mesma
                print( "A imagem " + nome_arquivo_comparado + " é do " + pessoa_comparada )
                # Salva imagem comparada e conhecida na pasta de comparações aprovadas
                save_image(imagem_comparada, nome_arquivo_comparado, nome_arquivo_conhecido, nome_database = "aprovadas")
            else: 
                print( "A imagem " + nome_arquivo_comparado + " não é do " + pessoa_comparada )
                # Salva imagem comparada e conhecida na pasta de comparações vetadas
                save_image(imagem_comparada, nome_arquivo_comparado, nome_arquivo_conhecido, nome_database = "vetadas")

    else: 
        # Caso a pessoa não seja encontrada na base de dados conhecida, ela não é comparada 
        # e um aviso é emitido no terminal
        dados_pessoa_sem_comparacao = dados_comparacao[pessoa_comparada]
        print("A pessoa " + pessoa_comparada + " não foi encontrada na base de dados conhecida.")
        for codigo_sem_comparacao, nome_arquivo_nao_comparado, imagem_sem_comparacao in dados_pessoa_sem_comparacao:
            print("A imagem com nome " + nome_arquivo_nao_comparado + " não teve uma pessoa encontrada na database correspondente")






