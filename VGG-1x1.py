# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:21:38 2019

@author: rodrigo.ferrari
"""

# check version of keras_vggface
import keras_vggface
# print version
print(keras_vggface.__version__)

import os
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import cosine
import cv2
from image_helpers import find_face, crop_face, open_crop_and_resize_face, rotate_image

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames,crop=True):
  if crop==True:
      # extract faces
      faces = [open_crop_and_resize_face(f) for f in filenames]
  else:
      faces = [pyplot.imread(f) for f in filenames]
      for i in range(0,len(faces)):
          faces[i] = cv2.resize(faces[i], (224, 224))
  i = 1
  while(i*10 < 360):
    faces = [rotate_image(faces[0], 10*i)] + faces
    i = i + 1
    cv2.imshow("Face", faces[0])
    cv2.waitKey(0)
  print(len(faces))

  # convert into an array of samples
  samples = asarray(faces, 'float32')
  # prepare the face for the model, e.g. center pixels
  samples = preprocess_input(samples, version=2)
  # create a vggface model
  model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
  # perform prediction
  yhat = model.predict(samples)
  return yhat

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
  # calculate distance between embeddings
  score = cosine(known_embedding, candidate_embedding)
  if score <= thresh:
    print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
  else:
    print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

# define filenames
os.chdir("known_people/rodrigo/")
filenames = ['rodrigo1.jpg', 'rodrigo2.jpg', 'rodrigo3.jpg', 'thiago1.jpg', 'Benedict Cumberbatch0.jpg']
# get embeddings file filenames
embeddings = get_embeddings(filenames,crop=True)
# define Aaron Paul
ids = embeddings[0]
# verify known photos of sharon
print('Positive Tests')
i = 0
while(i < 36):
  is_match(embeddings[i], embeddings[36])
  print(i)
  i = i + 1

i = 0
while(i < 36):
  print(i)
  is_match(embeddings[i], embeddings[37])
  i = i + 1

# verify known photos of other people
print('Negative Tests')
i = 0
while(i < 36):
  print(i)
  is_match(embeddings[i], embeddings[38])
  i = i + 1

i = 0
while(i < 36):
  print(i)
  is_match(embeddings[i], embeddings[39])
  i = i + 1
