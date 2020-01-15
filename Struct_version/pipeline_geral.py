import argparse
from conversion_functions import convert_pdf_to_image
from background_remove import background_remove
from orientation_correction import orientation_correction
from set_canonical_orientation import set_canonical_orientation
from crop_first_half import crop_first_half
from improve_img_quality import improve_img_quality
from text_OCR_cnh import cnh_roi_detection, ocr_text


def cnh_ocr_master(input_image, export=False, path_test_output=None , img_link= None):

    #### Carregando arquivo no formato pdf e convertendo para imagem
    pdf_image = convert_pdf_to_image(input_image)
    #### Retirando o background da imagem de entrada para isolar o documento da CNH
    bg_removed = background_remove(pdf_image)
    #### Eliminando angulação irregular do documento, caso exista
    corrected_image = orientation_correction(bg_removed)
    #### Validando orientação canônica da imagem, ou seja, orientação de leitura
    new_im = set_canonical_orientation(corrected_image)

    if new_im is None:
        print("|====================================================================|")
        print('[INFO] Imagem não concluiu processo de tratamento - OCR não realizado')
        print("|====================================================================|")
        roi_text = []
    else:
        ### Recortando apenas a primeira metade da CNH
        crop = crop_first_half(new_im)
        ### Melhorando a qualidade da imagem
        img = improve_img_quality(crop)
        ### Detectando e separando as regiões de interesse da CNH
        temp_draw, rec, bgr_temp_roi, bgr_temp_roi_otsu = cnh_roi_detection(img)
        #### Gerando OCR das regiões de interesse (ROIs)
        roi_text = ocr_text( rec , bgr_temp_roi, bgr_temp_roi_otsu, export, path_test_output , img_link)
    return roi_text
 
if __name__ == "__main__":   
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True,
    help="input pdf name")
    ap.add_argument("-ip", "--input_path", required=True,
    help="path to pdf files")
    ap.add_argument("-op", "--output_path", default=None,
    help="path to OCR results")
    ap.add_argument("-e", "--export", default="n",
    help="exporting results flag")
    args = vars(ap.parse_args())
    
    if args["export"] == "y":
        name = args["input_image"]
        name = name.replace(".pdf", ".png")
        export_flag = True
    else:
        name=None
        export_flag = False        

    input_image = args["input_path"]+args["input_image"]
    ocr_text = cnh_ocr_master( input_image, export_flag, args["output_path"], name )
    print("|====================================================================|")
    print('[INFO] Resultado do OCR')
    print("|====================================================================|")
    print( ocr_text )
    print("|====================================================================|")