# Module created to enconde faces of the dataset into embbeding
# Modifiers: Thiago Luis
# Last Edit: 23/10/2019

from imutils import paths
from matplotlib import pyplot
import face_recognition
import argparse
import pickle
import cv2
import os
from image_helpers import find_face, crop_face


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-db", "--database", required=True,
  help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", default="encodings.pickle",
  help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
  help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["database"]))
 
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
  # extract the person name from the image path
  print("[INFO] processing image {}/{}".format(i + 1,
    len(imagePaths)))
  name = imagePath.split(os.path.sep)[-2]
 
  # load the input image and convert it from BGR (OpenCV ordering)
  # to dlib ordering (RGB)
  image = pyplot.imread(imagePath)
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  boxes, resized = find_face(rgb)
  face = crop_face(rgb)

  # compute the facial embedding for the face
  encodings = face_recognition.face_encodings(face, boxes, 5)
  
  # loop over the encodings
  for encoding in encodings:
    # add each encoding + name to our set of known names and
    # encodings
    knownEncodings.append(encoding)
    knownNames.append(name)
  # load the known faces and embeddings

# Dump names 
data = {"encodings": knownEncodings, "names": knownNames }
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()