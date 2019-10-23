# Module created to develop auxiliary functions to deal with images
# Modifiers: Thiago Luis
# Last edit: 10/23/2019
import cv2
import face_recognition


def rotate_image(image, degrees):
  # get image height, width
  (h, w) = image.shape[:2]
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

  return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

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
      model="cnn")
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



