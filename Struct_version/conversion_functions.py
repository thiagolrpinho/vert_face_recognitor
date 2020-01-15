import cv2
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import argparse


def convert_pdf_to_image(pdf_name):
    image_object = convert_from_bytes(open(pdf_name, 'rb').read())
    image_matrix = np.array(image_object[0], np.uint8)
    image_matrix=cv2.cvtColor(image_matrix, cv2.COLOR_RGB2BGR) 
    return image_matrix


def pdf_to_image_routine():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    PDF_file = args["input"]

    # Store all the pages of the PDF in a variable
    print("[INFO] Carregando PDF e convertendo em imagem ...")
    pages = convert_from_path(PDF_file)
    # Grab only the first page and convert to image
    print("[INFO] Salvando primeira página do PDF como imagem")
    page = pages[0]
    filename = PDF_file.replace(".pdf", "")+".png"
    page.save(filename)
