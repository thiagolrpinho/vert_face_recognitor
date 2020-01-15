from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import numpy as np

def extract_face(image):
    # create the face detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(image)
    if len(results) == 0:
        print("[INFO] Nenhum rosto detectado.\n")
        face_coords = []
        return face_coords
    else:
        print("[INFO] Rosto detectado.\n")
        # extract the bounding box from the first face
        x1, y1, w, h = results[0]['box']
        x2, y2 = x1 + w, y1 + h
        face_coords = [(x1, y1), (x2, y2)]
        return face_coords