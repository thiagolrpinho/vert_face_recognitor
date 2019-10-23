# import the necessary packages
from imutils import paths
from matplotlib import pyplot
import face_recognition
import argparse
import pickle
import cv2
import os

def rotate_image(image, degrees):
  # get image height, width
  (h, w) = image.shape[:2
                      ]
  # calculate the center of the image
  center = (w / 2, h / 2)
  scale = 1.0
  
  # Perform the counter clockwise rotation holding at the center
  M = cv2.getRotationMatrix2D(center, degrees, scale)
  rotated = cv2.warpAffine(image, M, (h, w))

  return rotated

def resize_image(image, minimum_size ):
  scale_percent = 50
  width = int(image.shape[1] / 2 )
  height = int(image.shape[0] / 2)
  while( width > minimum_size and height > minimum_size):
    width = int(width / 2 )
    height = int(height / 2)

  dim = (width, height)

  return cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)

def find_face(image):
  resized = resize_image(image, 400)

  # detect the (x, y)-coordinates of the bounding boxes
  # corresponding to each face in the input image
  boxes = []
  degrees = 0
  while degrees < 360 and not boxes :
    # If none is found in a first try, it rotates the image counter clockwise
    # till it finds a face or has rotate it entirely
    degrees_rotation = 45
    resized = rotate_image(resized, degrees )
    boxes = face_recognition.face_locations(resized ,
      model=args["detection_method"])
    degrees = degrees + degrees_rotation
  
  return boxes, resized
  
def crop_face(image):
  ''' Receives a cv2 vector image and returns an image of a single face in it
  if none was found returns the original image '''

  boxes, resized = find_face(image)
  
  if( boxes ) :
    bottom_y, bottom_x, floor_y, floor_x = boxes[0]
    face = resized[ bottom_y:floor_y, floor_x:bottom_x ]
    return face
  return image


# MAIN PARA EXEMPLO E TESTES
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-db", "--database", required=True,
  help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
  help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
  help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
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
  encodings = face_recognition.face_encodings(face, boxes)
  
  # loop over the encodings
  for encoding in encodings:
    # add each encoding + name to our set of known names and
    # encodings
    knownEncodings.append(encoding)
    knownNames.append(name)
  # load the known faces and embeddings

print("[INFO] loading encodings...")
data = {'Names': knownNames
        ,'Encondings': knownEncodings
      }
 
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
names = []


# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(knownEncodings,
		encoding)
	name = "Unknown"

# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}
 
		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = knownNames[i]
			counts[name] = counts.get(name, 0) + 1
 
		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(resized, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(resized, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Image", resized)
cv2.waitKey(0)