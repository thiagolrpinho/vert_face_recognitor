import ntpath
import keras
from conversion_functions import convert_pdf_to_image
from background_remove import background_remove
from orientation_correction import orientation_correction
from set_canonical_orientation import set_canonical_orientation
from crop_first_half import crop_first_half
from improve_img_quality import improve_img_quality
from text_OCR_cnh import cnh_roi_detection, ocr_text
from suport_models import models_list


def create_dict(temp, roi_text, ind_image_quality):
    # Criando um dicinário para a interface com o resto da aplicação
    file_dict = {'file': temp.replace("./uploads/renach/", "")}

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

    image_quality_dict = {'sharpness': '', 'clarity': ''}
    if ind_image_quality != []:
        for i, key in enumerate(image_quality_dict.keys()):
            if len(ind_image_quality[i]) > 1:
                image_quality_dict[key] = ind_image_quality[i]
            else:
                image_quality_dict[key] = ind_image_quality[i][0]

    file_dict.update(roi_dict)
    file_dict.update(image_quality_dict)
    return file_dict


def cnh_ocr_master(pdf_list, export=False, path_test_output=None):

    list_renach_dicts = []
    edgeDetector, model_grabcut, model = models_list()
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

        renach_dict = create_dict(temp, roi_text, ind_image_quality)
        list_renach_dicts.append(renach_dict)

    keras.backend.clear_session()
    return list_renach_dicts
