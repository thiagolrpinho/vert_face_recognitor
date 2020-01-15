#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import math
from scipy import ndimage
from PIL import Image
import argparse
from framing_helper import print_image, trim

def orientation_correction(src_image):
    #print("[INFO] Importando imagem de entrada ...\n")

    #print("[INFO] Detectando angulação irregular ...\n")
    # Transform the image to gray scale
    img_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Detect image edges using Canny method
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)

    print_image(img_edges, title='Canny Image')

    # Detect lines in the image
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=50, maxLineGap=10)

    # Found angle of each of the lines
    angles = []

    for x1, y1, x2, y2 in lines[0]:
        # cv2.line(src_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    #print(angles)
    # Return the median of the detected angles.
    median_angle = np.median(angles)
    #print(median_angle)

    if median_angle != 0.0:
        #print("[INFO] Angulação irregular é de {} graus.\n"
        #      .format(median_angle))
        #print("[INFO] Corrigindo angulação irregular ...\n")
        # Rotate image using median angle

        img_rotated = ndimage.rotate(src_image, median_angle, cval=255)

        bg = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
        bg = Image.fromarray(bg)
        new_im = trim(bg)
        width, height = new_im.size

        #print("[INFO] Verificando se a imagem está deitada ...\n")
        if width > height:
            #print("[INFO]   Imagem deitada, rotacionando 90 graus ...\n")
            img_rotated = ndimage.rotate(img_rotated, 90, cval=255)
            bg = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
            bg = Image.fromarray(bg)
            new_im = trim(bg)
        new_im = np.asarray(new_im)
        new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2BGR)
    else:
        #print("[INFO] Imagem não possui angulação irregular ...\n")

        # Get image dimensions
        height, width = src_image.shape[:2]
        bg = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        print_image(bg, title='Pre Trimmed Image')
        bg = Image.fromarray(bg)
        new_im = trim(bg)

        #print("[INFO] Verificando se a imagem está deitada ...\n")
        if width > height:
            #print("[INFO]   Imagem deitada, rotacionando 90 graus ...\n")
            img_rotated = ndimage.rotate(src_image, 90, cval=255)
            bg = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
            bg = Image.fromarray(bg)
            new_im = trim(bg)
        new_im = np.asarray(new_im)
        new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2BGR)

    print_image(new_im, title="Orentation Corected image", shouldWait=True)
    return new_im


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
                    help="input image")
    ap.add_argument("-o", "--output_image", type=str, default="orientation_correction_output.png",
	help="path to output image")                
    args = vars(ap.parse_args())

    img_before = cv2.imread( args["input_image"] )
    corrected_image = orientation_correction(img_before)
    cv2.imwrite(args["output_image"], corrected_image)
