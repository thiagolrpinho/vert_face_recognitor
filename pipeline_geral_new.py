import argparse
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

def cnh_ocr_master(input_image, export=False, path_test_output=None):
    
    ocr_text_final = []
    ind_image_quality_final = []
    edgeDetector, model_grabcut, model = models_list()
    for temp in input_image:
    
        #### Carregando arquivo no formato pdf e convertendo para imagem
        pdf_image = convert_pdf_to_image(temp)
        #### Retirando o background da imagem de entrada para isolar o documento da CNH
        bg_removed = background_remove(pdf_image, edgeDetector, model_grabcut)
        #### Eliminando angulação irregular do documento, caso exista
        corrected_image = orientation_correction(bg_removed)
        #### Validando orientação canônica da imagem, ou seja, orientação de leitura
        new_im = set_canonical_orientation(corrected_image)

        if new_im is None:
            print("|====================================================================|")
            print('[INFO] Imagem não concluiu processo de tratamento - OCR não realizado')
            print("|====================================================================|")
            roi_text = []
            ind_image_quality = []
        else:
            ### Recortando apenas a primeira metade da CNH
            crop, ind_image_quality = crop_first_half(new_im)
            ### Melhorando a qualidade da imagem
            img = improve_img_quality(crop)
            ### Detectando e separando as regiões de interesse da CNH
            temp_draw, rec, bgr_temp_roi, bgr_temp_roi_otsu = cnh_roi_detection(img, model)
            #### Gerando OCR das regiões de interesse (ROIs)
            roi_text = ocr_text( rec , bgr_temp_roi, bgr_temp_roi_otsu, export, path_test_output , ntpath.basename(temp).replace(".pdf",".png") )
            if len(roi_text) != 8:
                roi_text = []
                print("|====================================================================|")
                print('[INFO] Resultado do OCR não gerou a quantidade de informação esperada, output invalidado')
                print("|====================================================================|")
            if len(ind_image_quality) != 2:
                ind_image_quality = []                
        ocr_text_final.append(roi_text)
        ind_image_quality_final.append(ind_image_quality)
        
    keras.backend.clear_session()
    return ocr_text_final , ind_image_quality_final
 
if __name__ == "__main__":   
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", nargs='+', required=True,
    help="list with input pdf names")
    ap.add_argument("-ip", "--input_path", required=True,
    help="path to pdf files")
    ap.add_argument("-op", "--output_path", default=None,
    help="path to OCR results")
    ap.add_argument("-e", "--export", default="n",
    help="exporting results flag")
    args = vars(ap.parse_args())
    
    if args["export"] == "y":
        export_flag = True
    else:
        export_flag = False        

    input_image = [ args["input_path"] + s for s in args["input_image"] ]
    ocr_text_final, ind_image_quality_final = cnh_ocr_master( input_image, export_flag, args["output_path"])
    print("|====================================================================|")
    print('[INFO] Medidas de qualidade da imagem:\n - Nível de desfoque\n - Nível de claridade')
    print("|====================================================================|")
    print( ind_image_quality_final )
    print("|====================================================================|")
    print('[INFO] Resultado do OCR')
    print("|====================================================================|")
    print( ocr_text_final )
    print("|====================================================================|")
