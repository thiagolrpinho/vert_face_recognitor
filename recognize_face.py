# import the necessary packages
import face_recognition
from matplotlib import pyplot
import argparse
import pickle
import cv2
from image_helpers import find_face, crop_face
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
  help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
  help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
  help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
 
# load the input image and convert it from BGR to RGB
print(args["image"])
image = pyplot.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes, resized = find_face(rgb)
face = crop_face(rgb)

encodings = face_recognition.face_encodings(face, boxes)
 
# initialize the list of names for each face detected
person_names = []
person_encodings = []

person_encodingsIdxs = [i for (i, b) in enumerate(data["names"]) if b == "rodrigo"]
for i in person_encodingsIdxs:
  person_encodings.append(data["encodings"][i])
  person_names.append(data["names"][i])

# loop over the facial embeddings
for encoding in encodings:
  # attempt to match each face in the input image to our known
  # encodings
  matches = face_recognition.compare_faces(person_encodings, encoding, 0.1)
  
  if True in matches:
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, person_names):
      # draw the predicted face name on the image
      cv2.rectangle(resized, (left, top), (right, bottom), (0, 255, 0), 2)
      y = top - 15 if top - 15 > 15 else top + 15
      cv2.putText(resized, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Image", resized)
cv2.waitKey(0)

