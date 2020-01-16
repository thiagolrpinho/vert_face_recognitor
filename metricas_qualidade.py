import cv2
import numpy as np

def metricas_qualidade (img):
    '''
    Recebe uma imagem em BGR e retorna as duas métricas de qualidade:
    1 - LapSNR
    2 - Claridade
    '''
    # Transformando em escala de cinza
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    ## Calculo laplace
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    # variacao
    lap_var = np.round(lap.var(), 2)
    # log da variacao
    log_lap = np.round(np.log(lap_var), 2)
    # media
    lap_mean = np.round(np.mean(lap),2)
    # SNR - signal-to-noise ratio
    snr = np.round(lap_mean/log_lap, 2)
    if snr <= 3:
        snr_str = "Baixa"
    elif snr > 3 and snr < 5:
        snr_str = "Media"
    else:
        snr_str = "Alta"
    # Claridade
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
    clar = np.median(hsv/255)
    if clar <= 0.55:
        clar_str = "Escura"
    elif clar > 0.55 and clar < 0.7:
        clar_str = "Media"
    elif clar >=0.7 and clar < 0.85:
        clar_str = "Alta"
    else:
        clar_str = "Muito Alta"

    return ([snr_str, clar_str])

