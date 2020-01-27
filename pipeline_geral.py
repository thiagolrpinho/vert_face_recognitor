import keras
import pickle
from conversion_functions import convert_pdf_to_image
from background_remove import background_remove
from orientation_correction import orientation_correction
from set_canonical_orientation import set_canonical_orientation
from crop_first_half import crop_first_half
from improve_img_quality import improve_img_quality
from text_OCR_cnh import cnh_roi_detection, ocr_text
from suport_models import models_list
from similarity_validation import similarity_validation


def create_roi_dict(roi_text):
    # criando dicionario de roi
    roi_dict = {
        'name': '', 'rg': '', 'cpf': '', 'birth_date': '', 'parents': '',
        'renach_number': '', 'expire_date': '', 'first_renach_date': ''}
    if roi_text != []:
        for i, key in enumerate(roi_dict.keys()):
            roi_len = len(roi_text[i])
            if roi_len == 0:
                roi_dict[key] = ''
            elif roi_len == 1:
                roi_dict[key] = roi_text[i][0]
            else:
                roi_dict[key] = roi_text[i]
    return roi_dict


def create_imgage_quality_dict(ind_image_quality):
    # criando dicionario de qualidade de imagem
    image_quality_dict = {'sharpness': '', 'clarity': ''}
    if ind_image_quality != []:
        for i, key in enumerate(image_quality_dict.keys()):
            img_len = len(ind_image_quality[i])
            if img_len == 0:
                image_quality_dict[key] = ''
            elif img_len == 1:
                image_quality_dict[key] = ind_image_quality[i][0]
            else:
                image_quality_dict[key] = ind_image_quality[i]
    return image_quality_dict


def create_score_geral_dict(score_geral):
    # criando dicionario de score
    score_geral_dict = {'score': '', 'result': ''}
    if score_geral != []:
        for i, key in enumerate(score_geral_dict.keys()):
            score_geral_dict[key] = str(score_geral[i]).replace(".", ",")
    return score_geral_dict


def create_dict(img_id, roi_text, ind_image_quality, score_geral):
    # Criando um dicinário para a interface com o resto da aplicação
    file_dict = {'file': img_id}
    roi_dict = create_roi_dict(roi_text)
    image_quality_dict = create_imgage_quality_dict(ind_image_quality)
    score_geral_dict = create_score_geral_dict(score_geral)
    file_dict.update(roi_dict)
    file_dict.update(image_quality_dict)
    file_dict.update(score_geral_dict)
    return file_dict


def cnh_ocr_master(pdf_list, export=False, path_test_output=None):

    # Definindo lista
    list_renach_dicts = []
    # Carregando Modelo
    edgeDetector, model_grabcut, model = models_list()
    # Carregando arquivos de banco de fotos
    filehandler = open("./ids_banco_manual", "rb")
    ids_banco_manual = pickle.load(filehandler)
    filehandler.close()
    filehandler = open("./info_banco_manual", "rb")
    info_banco_manual = pickle.load(filehandler)
    filehandler.close()

    for temp in pdf_list:
        # Carregando arquivo no formato pdf e convertendo para imagem
        pdf_image = convert_pdf_to_image(temp)
        # Retirando o background da imagem de entrada
        # para isolar o documento da CNH
        bg_removed = background_remove(pdf_image, edgeDetector, model_grabcut)
        # Eliminando angulação irregular do documento, caso exista
        corrected_image = orientation_correction(bg_removed)
        # Validando orientação canônica da imagem,
        # ou seja, orientação de leitura
        new_im = set_canonical_orientation(corrected_image)

        if new_im is None:
            print("|===================================\
=================================|")
            print('[INFO] Imagem não concluiu \
processo de tratamento - OCR não realizado')
            print("|====================================\
================================|")
            roi_text = []
            ind_image_quality = []
        else:
            # Recortando apenas a primeira metade da CNH
            crop, ind_image_quality = crop_first_half(new_im)
            # Melhorando a qualidade da imagem
            img = improve_img_quality(crop)
            # Detectando e separando as regiões de interesse da CNH
            temp_draw, rec, bgr_temp_roi,\
                bgr_temp_roi_otsu = cnh_roi_detection(img, model)
            # Gerando OCR das regiões de interesse (ROIs)
            roi_text = ocr_text(rec, bgr_temp_roi, bgr_temp_roi_otsu, export,
                                path_test_output, None)
            if len(roi_text) != 8:
                roi_text = []
                print("|====================================\
================================|")
                print('[INFO] Resultado do OCR não gerou a \
quantidade de informação esperada, output invalidado')
                print("|=====================================\
===============================|")
            if len(ind_image_quality) != 2:
                ind_image_quality = []

        img_id = temp.replace("./uploads/renach/", "")
        id_predito = [img_id]
        info_predito = [roi_text]
        score_geral, _ = similarity_validation(
            ids_banco_manual, info_banco_manual, id_predito, info_predito)
        renach_dict = create_dict(
            img_id, roi_text, ind_image_quality, score_geral[0])
        list_renach_dicts.append(renach_dict)

    keras.backend.clear_session()
    return list_renach_dicts
