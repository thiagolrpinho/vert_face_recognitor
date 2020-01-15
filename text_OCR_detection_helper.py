# Módulo criado para definir funções auxiliares para os processos
# de detecção de texto e OCR

from keras.models import Sequential,load_model
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Activation, Flatten, Input
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np 
import cv2
import pytesseract
from PIL import Image
from framing_helper import show_img

def predict_model():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN
    
#### Funcoes de perda da rede necessárias para ela funcionar
SMOOTH = 1.0
def dice_coef(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + SMOOTH) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def np_dice_coef(y_true, y_pred):
    y_true_flat = y_true.flat[:]
    y_pred_flat = y_pred.flat[:]
    intersection = np.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + SMOOTH) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + SMOOTH)

def np_dice_coef_loss(y_true, y_pred):
    return 1.0 - np_dice_coef(y_true, y_pred)

#### Funcao para retirar linhas horizontais e verticais da CNH
def retirar_linhas_horVert(image):
    # load the example image and convert it to grayscale
    # inicio = time.time()
    d_y_hor_rec = 10
    d_x_hor_rec = 80
    d_x_vert_rec = 5
    d_y_vert_rec = 80
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show_img(gray)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # show_img(cv2.resize(otsu, (700, 900)))
    # operacao retangulo vertical
    rectVertKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d_x_vert_rec, d_y_vert_rec))
    gradX_vert = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, rectVertKernel)
    thresh_vert = cv2.threshold(gradX_vert, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # show_img(cv2.resize(thresh_vert, (700, 900)))
    # operacao retangulo vertical
    rectHorKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d_x_hor_rec, d_y_hor_rec))
    gradX_hor = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, rectHorKernel)
    thresh_hor = cv2.threshold(gradX_hor, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # show_img(cv2.resize(thresh_hor, (900, 700)))
    # criando mascara com resultado dos kernels vert e hor
    soma_kernels = cv2.bitwise_and(thresh_vert, thresh_vert, mask = thresh_hor)
    # show_img(cv2.resize(soma_kernels, (900, 700)))
    ero = cv2.erode(soma_kernels, None, iterations=5)
    # show_img(cv2.resize(ero, (900, 700)))
    # Fazendo operacao de mascara
    image[np.where(ero == 0)] = 255
    # show_img(cv2.resize(image, (900, 700)))
    return image

#### Funcao para detecção de texto
def img_to_text (roi, roi_otsu, index_regiao):
    #### Flags para resize
    if (index_regiao == 2) or (index_regiao == 3):
        index_resize = 5
    elif (index_regiao == 4) or (index_regiao == 0):
        index_resize = 4
    else:
        index_resize = 3
    #############
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)[:,:,1]
    hls = hls[hls <= 250]/255
    # print("Media = " + str(hls.mean()))
    # print("STD = " + str(hls.std()))
    if (hls.mean() <= 0.55):
        roi = adaptive_gamma_correction(roi)
        # temp,_,_ = automatic_brightness_and_contrast(temp)
        # show_img(roi)
    ##############
    #### Processando as regioes
    if (index_regiao == 5) or (index_regiao == 6):
        ### Resize
        temp = cv2.resize(roi, (int(roi.shape[1]*index_resize), int(roi.shape[0]*index_resize)))
        # show_img(temp)
        ### Escala de cinza
        gray = cv2.cvtColor(temp.copy(), cv2.COLOR_BGR2GRAY)
        # show_img(gray)
        ### Filtro de mediana
        median = cv2.medianBlur(gray.copy(), ksize = 5)
        # show_img(median)
        ### Equalizacao por histograma
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6,6))
        clahe = clahe.apply(median.copy())
        # show_img(clahe)
        ### Binarizacao
        otsu = cv2.threshold(clahe.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        # show_img(otsu)
    else:
        ### Resize
        roi = cv2.resize(roi, (int(roi.shape[1]*index_resize), int(roi.shape[0]*index_resize)))
        roi_otsu = cv2.resize(roi_otsu, (int(roi_otsu.shape[1]*index_resize), int(roi_otsu.shape[0]*index_resize)))
        # show_img(roi)
        ## Escala de cinza
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # show_img(gray)
        if (index_regiao == 1) or (index_regiao == 2):
            ### Filtro mediana
            gray = cv2.medianBlur(gray, ksize = 5)
            # show_img(gray)
        ### Binarizacao
        otsu = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        # show_img(otsu)
        ### Removendo ruidos de background com imagem geral binarizada
        otsu[np.where(roi_otsu == 255)] = 255
        # show_img(otsu)
    ### Adicionando padding branco
    otsu = cv2.copyMakeBorder(otsu, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, 255)
    # show_img(otsu)
    ### Operacao morfologica para remocao de ruidos
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    # show_img(opening)
    ### Transformando em imagem "PIL"
    pil_img = Image.fromarray(opening)
    ### Aplicando OCR
    config = ("-l por --oem 1 --psm 6")
    text = pytesseract.image_to_string(pil_img, config=config)
    
    #### Retirando caracteres especiais
    ### lista negra de caracteres
    blacklist = "[])}.€-|—~^=«,\({#*:!@#$%&*,;:»_ªº?@<>+]”“"
    text_split = text.split("\n")
    # Retirando todos os characters de forma segura (sem erro)
    for index_ts, t_s in enumerate(text_split):
        for char in blacklist:
            text_split[index_ts] = text_split[index_ts].replace(char, "")
    # Retirando char especial = ''
    text_final = []
    for char in text_split:
        if (char == ''):
            continue
        else:
            text_final.append(char)
    # Caso "text_final" nao tenha valor nenhum 
    # Fazer essa verificacao
    if text_final:
        # Retirando manualmente alguns outros caracteres especiais
        text_final = retirando_chars_especiais(text_final, index_regiao)
    return text_final

def retirando_chars_especiais(text_final, index_regiao):
    if index_regiao == 0:
        ### Caso text_final tenha 2 posições ou mais 
        if (len(text_final) >= 2):
            ## Retirar a primeira que costuma ser ruido
            text_final.pop(0)
    elif index_regiao == 1:
        ### Caso text_final tenha 2 posições ou mais 
        if (len(text_final) >= 2):
            ## Costuma ter ruido a partir da segunda posicao da lista
            text_final = [text_final[0]]
    elif index_regiao == 4:
        ### Caso text_final tenha 2 posições ou menos
        if (len(text_final[0]) <= 2):
            ## Retirar a primeira que costuma ser ruido
            text_final.pop(0)
    elif ((index_regiao == 2) or (index_regiao == 3) or (index_regiao >= 5)):
        if (len(text_final) >= 2):
            ### Organizar para as maiores strings ficarem por ultimo, assim e so pegar ela
            temp = list(zip(text_final, [len(x) for x in text_final]))
            temp.sort(key = lambda tup: tup[1])
            text_final = [temp[len(temp)-1][0]]
        ### Dando replace em alguns casos necessarios
        text_final = [text_final[0].replace("S", "5")]
        text_final = [text_final[0].replace(" ", "")]
    return text_final

#### Funcao para organizar os contornos de maneira "cima -> baixo" e "esquerda -> direita"
def organizar_contornos(rec = None):
    #### De cima pra baixo
    rec.sort(key = lambda tup: tup[1])
    
    #### Esquerda para direita _1
    temp = rec[2:4].copy()
    temp.sort(key = lambda tup: tup[0])
    rec[2:4] = temp

    #### Esquerda para direita _2
    temp = rec[5:].copy()
    temp.sort(key = lambda tup: tup[0])
    rec[5:] = temp

    return rec

#### Funcao para melhorar brilho e contraste atraves do parâmetro gamma de maneira automatica
def adaptive_gamma_correction (img):
    ### Transformar para hls
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hls_v = hls[:,:,1]/255
    # show_img(hls)
    ########################## METODO ##########################
    # I_out = c * pow(I_in, gamma)
    # O objetivo e achar os parametros "c" e "gamma"
    ### Parametros para classificacao
    mean = hls_v[hls_v <= 0.98].mean() # 0.98 == 250
    std = hls_v[hls_v <= 0.98].std()
    tal = 3 # arbitrario
    D = (mean + 2*std) - (mean - 2*std)
    ### Arvore de decisao
    if (D <= (1/tal)): # Contraste baixo
        if (mean >  0.55): # Imagem clara
            c = 1
            gamma = -np.log2(std)
            new_hls_v = np.clip(c * np.power(hls_v, gamma), 0, 1)*255
            new_hls_v = new_hls_v.astype(int)
        else: # Imagem escura
            gamma = -np.log2(std)
            power_gamma = np.power(hls_v, gamma)
            temp = power_gamma / ( (power_gamma) + ( (1 - power_gamma) * (np.power(mean, gamma) ) ) )
            new_hls_v = np.clip(temp, 0, 1)*255
            new_hls_v = new_hls_v.astype(int)
    else:
        if (mean > 0.55): # Imagem clara
            gamma = np.exp( (1 - (mean + std)) / 2)
            c = 1
            new_hls_v = np.clip(c * np.power(hls_v, gamma), 0, 1)*255
            new_hls_v = new_hls_v.astype(int)
        else: # Imagem escura
            gamma = np.exp( (1 - (mean + std)) / 2) + 0.1
            power_gamma = np.power(hls_v, gamma)
            temp = power_gamma / ( (power_gamma) + ( (1 - power_gamma) * (np.power(mean, gamma) ) ) )
            new_hls_v = np.clip(temp, 0, 1)*255
            new_hls_v = new_hls_v.astype(int)
    ##############################################################################
    #### Voltando com imagem original
    ori_hls = hls.copy()
    ori_hls[:,:,1] = new_hls_v
    # show_img(ori_hls)
    ori_bgr = cv2.cvtColor(ori_hls, cv2.COLOR_HLS2BGR)
    # show_img(ori_bgr)
    return ori_bgr