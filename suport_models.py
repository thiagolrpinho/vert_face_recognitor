import cv2
from PIL import Image
import numpy as np
import pickle
from keras.models import Sequential,load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Activation, Flatten, Input
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from text_OCR_detection_helper import dice_coef,dice_coef_loss,np_dice_coef,np_dice_coef_loss

def models_list():
    #print("[INFO] Carregando modelo de detecção de bordas ...\n")
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("assets/model.yml")

    #print("[INFO] Carregando modelo GrabCut ...\n")
    model_grabcut = pickle.load(open("assets/model_grabcut_pdf2imgRoberto.vrt", "rb") )
            
    #### Carregando modelo para segmentação das áreas de texto da CNH
    model = load_model("assets/modelo_segmentacao.h5", custom_objects={'dice_coef_loss': dice_coef_loss, "dice_coef": dice_coef})

    return edgeDetector, model_grabcut, model