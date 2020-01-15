from keras.models import Sequential,load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Activation, Flatten, Input
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np 
import cv2
import os
import pytesseract
from PIL import Image
from framing_helper import show_img
import argparse
from text_OCR_detection_helper import dice_coef,dice_coef_loss,np_dice_coef,np_dice_coef_loss, \
                                        retirar_linhas_horVert,img_to_text,retirando_chars_especiais, \
                                        organizar_contornos,adaptive_gamma_correction

def cnh_roi_detection(bgr):

    #### Carregando modelo para segmentação das áreas de texto da CNH
    model = load_model("assets/modelo_segmentacao.h5", custom_objects={'dice_coef_loss': dice_coef_loss, "dice_coef": dice_coef})
    
    #### Retirando linhas horizontais e verticais
    bgr_without_blanklines = retirar_linhas_horVert(image = cv2.resize(bgr, (int(2688*3/4), (int(1956*3/4)))))
    bgr_resize = cv2.resize(bgr_without_blanklines, (bgr.shape[1], bgr.shape[0]))
    # show_img(cv2.resize(bgr_resize, (900, 600)))
    
    #### Predicao neural
    ### Escala de cinza e dando resize
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_pred = gray.copy()/255
    gray_resize = cv2.cvtColor(bgr_resize, cv2.COLOR_BGR2GRAY)
    resize_gray = cv2.resize(gray_pred, (128,128))
    # show_img(resize_gray)
    ### Gerando mascara de segmentacao
    pred = model.predict(resize_gray.reshape(1, resize_gray.shape[0], resize_gray.shape[1], 1))
    img_pred = pred[0,:,:,0]
    # show_img(img_pred)
    ### Aplicando threshold
    img_pred[img_pred>0.5] = 1
    img_pred[img_pred<=0.5] = 0
    ### Resize para tamanho normal da mascara
    img_pred_res = cv2.resize(img_pred, (bgr.shape[1], bgr.shape[0]))
    # show_img(img_pred_res)
    ### Aplicando novamente threshold para não ficar regioes de penumbra
    img_pred_res[img_pred_res>=0.5] = 1
    img_pred_res[img_pred_res<0.5] = 0

    #### Segmentando somente a parte necessaria da imagem original
    bgr_temp_cnt = bgr.copy()
    bgr_temp_cnt[np.where(img_pred_res == 0)] = 0
    bgr_temp_cnt[np.where(gray_resize == 255)] = 0
    # show_img(cv2.resize(bgr_temp_cnt, (900, 600)))

    bgr_temp_roi = bgr.copy()
    bgr_temp_roi[np.where(img_pred_res == 0)] = 255
    bgr_temp_roi[np.where(gray_resize == 255)] = 255
    # show_img(cv2.resize(bgr_temp_roi, (900, 600)))
    
    bgr_temp_roi_gray = cv2.cvtColor(bgr_temp_roi, cv2.COLOR_BGR2GRAY)
    bgr_temp_roi_otsu = cv2.threshold(bgr_temp_roi_gray, 127, 255, cv2.THRESH_BINARY)[1]
    # show_img(cv2.resize(bgr_temp_roi_otsu, (900, 600)))

    #### Achando contornos
    ### Escala de cinza na predicao
    gray_temp = cv2.cvtColor(bgr_temp_cnt, cv2.COLOR_BGR2GRAY)    
    # show_img(gray_temp)
    ### Achando contornos
    contours,hierarchy = cv2.findContours(image = gray_temp.copy(), mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
    ### Organizando contornos    
    ## Somente area percentual maior que 0.5
    cnt_real = []
    rec = []
    for cnt in contours:
        # print(cv2.contourArea(cnt))
        area_total = bgr_temp_cnt.shape[0]*bgr_temp_cnt.shape[1]
        area_roi = cv2.contourArea(cnt)
        perc_area = 100*(area_roi/area_total)
        if perc_area > 0.5:
            cnt_real.append(cnt)
            rec.append(cv2.boundingRect(cnt))
    ### Organizar pela ordem dos retangulos
    rec = organizar_contornos(rec = rec)
    # show_img(gray_temp)
    # show_img(bgr_resize)
    # show_img(bgr_temp_roi)
    # show_img(bgr_temp_cnt)
    
    #### Variaveis temporarias para modificacao
    temp_draw = bgr_temp_cnt.copy()
    
    #### Loop para cortar as imagens e gerar os textos
    for cnt in rec:
        ### Gerando os retangulos do contorno
        x,y,w,h = cnt
        ### Desenhando o contorno
        _ = cv2.rectangle(temp_draw, (x, y), (x+w, y+h), (0,255,0), 2)

    return temp_draw, rec, bgr_temp_roi, bgr_temp_roi_otsu
   
def ocr_text( rec , bgr_temp_roi, bgr_temp_roi_otsu, export=False, path_test_output=None , img_link= None):
    if (export==True):
        #### Criando diretorio para salvar imagens
        if not(os.path.exists(path_test_output + img_link.replace(".png", ""))):
            os.mkdir(path_test_output + "/" + img_link.replace(".png", ""))
    
    #### Variaveis temporarias para modificacao
    roi_text = []

    #### Loop para cortar as imagens e gerar os textos
    for index_regiao, cnt in enumerate(rec):
        ### Gerando os retangulos do contorno
        x,y,w,h = cnt
        ### Gerando os cortes
        croped = bgr_temp_roi[y:y+h, x:x+w].copy()
        # show_img(croped)
        croped_otsu = bgr_temp_roi_otsu[y:y+h, x:x+w].copy()
        if (export==True):
            ### Salvando a imagem recortada
            _ = cv2.imwrite(path_test_output + img_link.replace(".png", "/") + img_link.replace(".png", "") + "_{}.png".format(index_regiao), croped)
        ### Gerando texto
        roi_text.append(img_to_text(roi = croped, roi_otsu = croped_otsu, index_regiao = index_regiao))

    if (export==True):
        #### Salvando txt com os dados do arquivo
        f = open(path_test_output + img_link.replace(".png", "/") + img_link.replace(".png", ".txt"), "w+")
        for line in roi_text:
            if (len(line) >= 1):
                temp = []
                if not(line[0] == ' '):
                    # Agrupando os dados sem ' '
                    for text in line:
                        if not(text == ' '):
                            temp.append(text)
                    temp = temp + ["\n"]
                    _ = f.write("|".join(temp))
        f.close()
    return roi_text

if __name__ == "__main__":    
### Import image
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
                    help="input pdf name")
    ap.add_argument("-p", "--path_ocr", required=True,
                    help="path to OCR results")
    args = vars(ap.parse_args())	
    bgr = cv2.imread(args["input_image"])
    temp_draw, rec, bgr_temp_roi, bgr_temp_roi_otsu = cnh_roi_detection(bgr)
    
    #### Salvando a imagem com contorno
    _ = cv2.imwrite("CNH_text_roi_detected.png", temp_draw) 
    #### Gerando OCR
    ocr_text( rec , bgr_temp_roi, bgr_temp_roi_otsu, True, args["path_ocr"] , args["input_image"])
    