#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import argparse
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from framing_helper import trim 
from extract_face import extract_face

def set_canonical_orientation(img_before):

    #print("[INFO] Importando imagem de entrada ...\n")

    bg = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
    new_im = Image.fromarray(bg)
    width, height = new_im.size
    new_im = np.asarray(new_im)

    #print("[INFO] Iniciar processo para detecção de rosto no documento "
    #      "e confirmar orientação canônica ...\n")
    cont = 1
    while cont <= 4:
        #print("========")
        #print("[INFO] {}ª Verificação.".format(cont))
        #print("========\n")
        face = extract_face(new_im)
        if len(face) > 0:
            x1, y1 = face[0]
            x2, y2 = face[1]
            face_area = (x2-x1)*(y2-y1)
            prop_face_area = face_area/(height*width)*100
            if height > width:
                if y1 < 0.5*height and y2 < 0.5*height and x1 < 0.5*width and x2 \
                        < 0.5*width and prop_face_area > 1:
                    #print(
                    #    "[INFO] Verificada a orientação canônica do documento.\n")
                    break
                #else:
                    #print("[INFO] Coordenadas do rosto detectado não estão na "
                    #      "área esperada.\n")
            else:
                if x1 < 0.5*width and x2 < 0.5*width and prop_face_area > 2:
                    #print(
                    #    "[INFO] Verificada a orientação canônica do documento.\n")
                    break
                #else:
                    #print(
                    #    "[INFO] Coordenadas do rosto detectado não estão na "
                    #    "área esperada.\n")
        #print("[INFO] Rotacionando o documento e buscando novamente o rosto ...\n")
        img_before = ndimage.rotate(img_before, 90, cval=255)
        bg = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        bg = Image.fromarray(bg)
        new_im = trim(bg)
        width, height = new_im.size
        new_im = np.asarray(new_im)
        cont = cont + 1
        if cont == 5:
            print("========")
            print("[INFO] Não foi possível validar a orientação correta "
                  "do documento.")
            print("========\n")

    if cont < 5:
        new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2BGR)
        return new_im

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
                    help="path to input image")
    ap.add_argument("-o", "--output_image", type=str,
                    default="canonical_orientation_output.png",
                    help="path to output image")
    args = vars(ap.parse_args())

    img_before = cv2.imread(args["input_image"])
    new_im = set_canonical_orientation(img_before)
    cv2.imwrite(args["output_image"], new_im)
