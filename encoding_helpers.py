# Módulo auxiliar para gerenciar as funções relaciondas a converter imagens em códigos de incorporação
# Modifiers: Thiago Luis
# Last edit: 2019/10/29

from imutils import paths
import os
from image_helpers import find_face, crop_face, open_crop_and_resize_face, rotate_image
from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine


def get_embeddings( database_folder_name, crop=True ):
  ''' Recebe uma lista de nomes de arquivos para serem abertos. Caso crop = True, ele antes de calcular
  o embedding tenta encontrar o rosto e retorna uma imagem com somente ele. 
  Retorna uma lista de códigos de incorporação e uma lista dos nomes correspondentes desses códigos'''
  # Pega o caminho para a database passada no argumento
  print("[INFO] Contando rostos em " + database_folder_name + " ...")
  imagePaths = list(paths.list_images( database_folder_name ))
  faces = []
  filenames = []
  person_names = []
  images = []
  # Itera sobre os caminhos das imagens
  for (i, imagePath) in enumerate(imagePaths):
    # Extrai a face da pessoa na imagem, caso haja
    print("[INFO] Processando imagem {}/{}".format(i + 1,
      len(imagePaths)))

    filename = imagePath.split(os.path.sep)[-1]
    person_name = imagePath.split(os.path.sep)[-2]
    filenames.append(filename)
    person_names.append(person_name)
    
    if crop == True:
      # Se os rostos precisarem ser cortados das fotos, ele o faz.
      face_nparray, face_image = open_crop_and_resize_face(imagePath)
      faces.append(face_nparray)
      images.append(face_image)
    else:
      # Caso contrário ele somente abre e muda para o tamanho adequado.
      face_nparray = pyplot.imread(imagePath)
      face_nparray = cv2.resize(face_nparray, (224,224) )
      faces.append(face_nparray)
      images.append(face_image)

    #  Converte em um array de samples
  samples = asarray(faces, 'float32')
  # Faz os preprocessamentos necessários nas imagens para elas entrarem no modelo. Ex: Centraliza o rosto
  samples = preprocess_input(samples, version=2)
  # Cria um modelo já treinado do VGGFace
  model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
  # Aplica o modelo sobre as imagens e retorna um vetor de códigos de incorporação
  embeddings = model.predict(samples)

  return embeddings, person_names, filenames, images


def is_match(known_embedding, candidate_embedding, thresh = 0.5):
  # Retorna verdadeiro se estiver abaixo do limiar de diferença ou falso se for acima
  # Dado que os códigos são descritos como uma coordenada em 128 dimensões
  # Calculamos a distância entre as duas imagens como pontos
  score = cosine(known_embedding, candidate_embedding)
  if score <= thresh:
    print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    return True
  else:
    print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return False

def store_codes_with_names( embbedings, person_names, filenames, images):
  data = {}
  for embbeding, person_name, filename, image in zip( embbedings, person_names, filenames, images ):
    # Adiciona cada código e nome de arquivo para cada nome de pessoa
    if person_name in data:
      data[person_name].append( [embbeding, filename, image] )
    else: 
      data[person_name] = [[embbeding, filename, image]]
  return data