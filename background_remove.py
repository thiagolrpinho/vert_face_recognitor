from PIL import Image
import cv2
import os
import numpy as np
import argparse
import pickle
from framing_helper import filterOutSaltPepperNoise, findSignificantContour, \
                           trim, document_type
from conversion_functions import convert_pdf_to_image
from suport_models import models_list

def background_remove(src_image, edgeDetector, model_grabcut):
    #print("[INFO] Importando imagem de entrada ...\n")

    #print(
    #    "[INFO] Iniciando processo de eliminação de ruído no background ...\n")
    # First some gaussian blur to reduce noise.
    blurred = cv2.GaussianBlur(src_image, (5, 5), 0)

    # Edge detection using a structured forest ML approach from an opencv
    # contribution module.
    blurred_float = blurred.astype(np.float32) / 255.0

    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)
    contour = findSignificantContour(edges_8u)

    # Draw the contour on the original image
    contourImg = np.copy(src_image)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2,
                     cv2.LINE_AA, maxLevel=1)

    # Use grabcut algorithm to get the exact edges. However grabcut requires a
    # hint on sure foreground, sure background and probable foregorund areas.
    # Normally this information is provided manually (i.e. a person needs to
    # mark these areas). However we can approximate it automatically by
    # offsetting the contour to a “reasonably safe” margin.
    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255

    #print("[INFO] Realizando processo de recorte apenas da área do documento "
    #      "e separando do background ...\n")
    # run grabcut
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src_image, trimap, rect, model_grabcut["bgdmodel"],
                model_grabcut["fgdmodel"], 1, cv2.GC_EVAL_FREEZE_MODEL)
    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
        ).astype('uint8')

    contour2 = findSignificantContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255
    alpha[alpha > 255] = 255
    alpha = alpha.astype(float)

    foreground = np.copy(src_image).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv2.add(foreground, background)

    cv2.imwrite("output.png", cutout)

    #print("[INFO] Eliminando bordas externas e isolando a área do "
    #      "documento ...\n")
    bg = Image.open("output.png")  # The image to be cropped
    new_im = trim(bg)

    #print("[INFO] Salvando imagem tratada ...\n")
    numpy_image = np.array(new_im)
    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    os.remove("output.png")
    return opencv_image


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
                    help="input to pdf of CNH")
    ap.add_argument("-o", "--output_image", type=str,
                    default="bg_removed_output.png",
                    help="path to output image")
    args = vars(ap.parse_args())
    
    edgeDetector, model_grabcut = models_list()[0:1]
    pdf_image = convert_pdf_to_image(args["input_image"])
    bg_removed = background_remove(pdf_image, edgeDetector, model_grabcut)

    cv2.imwrite(args["output_image"], bg_removed)
