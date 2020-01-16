#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import argparse
from PIL import Image, ImageChops
from framing_helper import trim
from metricas_qualidade import metricas_qualidade 

def crop_first_half(image):
    #print("[INFO] Importando imagem de entrada ...\n")
    # Import image

    # Get image dimensions
    #print("[INFO] Dimensões da imagem de entrada:\n")
    height, width = image.shape[:2]
    #print("[INFO] Altura: {}".format(height))
    #print("[INFO] Largura: {}\n".format(width))

    #print("[INFO] Verificando a razão entre largura e altura do documento ...\n")
    #print("[INFO] A razão largura:altura padrão para o documento aberto deve ser 85:120 ~ {} \n".format(round(85/120,2)) )
    #print("[INFO] A razão largura:altura da imagem de entrada: {}:{} ~ {}\n".format( width,height,round(width/height,2)) )

    new_height = int( (60/85)*width )
    new_height = int(new_height*1.05)
    crop = image[ 0:new_height, 0:width ]
    ## Remove extra white space left from previus process and repeat crop rotine
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(crop)
    crop = trim(crop)
    crop = np.asarray(crop)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    height, width = crop.shape[:2]
    new_height = int( (60/85)*width )
    #new_height = int(new_height*1.03)
    new_height = int(new_height*1.05)
    crop = crop[ 0:new_height, 0:width ]
    ind_image_quality = metricas_qualidade(crop)
    return crop, ind_image_quality

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
    help="path to input image")
    ap.add_argument("-o", "--output_image", type=str, default="crop_half_output.png",
    help="path to output image")
    args = vars(ap.parse_args())

    image = cv2.imread( args["input_image"] )
    crop, ind_image_quality = crop_first_half(image)
    cv2.imwrite( args["output_image"] , crop)

