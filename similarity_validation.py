import textdistance
import unicodedata
import numpy as np

def remove_acentos(palavra):
    processamento_2 = unicodedata.normalize("NFD", palavra)
    processamento_2 = processamento_2.encode("ascii", "ignore")
    processamento_2 = processamento_2.decode("utf-8")
    return processamento_2

def ocr_result_treatment(x):

    ocr_text_final_trat = []
    
    for i in x:
        if len(i) == 0:
            temp = [ ]
        else:        
            i_temp = []
            for j in i:
                j = " ".join(j)
                j = j.strip()
                blacklist = '[])}.€-|—~^=«,\({#*:!@#$%&*,;:»_ªº?@<>+]”“""'
                for char in blacklist:
                    j = j.replace(char, "").strip()
                j = " ".join(j.split())
                j = j.upper()
                j = remove_acentos(j)
                i_temp.append(j)

            ## Aplica trim na string
            rg = i_temp[1].strip()
            ## Quebra a string em espaços e salva como lista
            rg = rg.split(" ")
            ## Elimina listas com tamanho 0, caso haja
            rg = [x for x in rg if len(x) != 0]
            if len(rg) == 0:
                rg = ''
            else:
                rg = rg[0]
            i_temp[1] = rg
            temp = i_temp
        ocr_text_final_trat.append(temp)
    
    return ocr_text_final_trat

def sortlist_by_index(index, rlist):
    sort_rlist = [x for _,x in sorted(zip(index, rlist ))]
    sort_index = sorted(index)
    return sort_index, sort_rlist

def matching_lists(id_real, real, id_pred, pred):
    ## achar interseccao
    inter = list( set( id_real  ).intersection( id_pred ) )
    
    ## achar index de inter em cada id
    index_real = np.where( np.isin( id_real , inter  ))[0].tolist()
    index_pred = np.where( np.isin( id_pred , inter  ))[0].tolist()
    
    ## filtrar todas as listas por esses indexes
    id_real_filt = [id_real[i] for i in index_real] 
    real_filt = [real[i] for i in index_real] 
    id_pred_filt = [id_pred[i] for i in index_pred] 
    pred_filt = [pred[i] for i in index_pred] 
    
    ## ordenar as listas pelo id
    s_index_real, s_real = sortlist_by_index(id_real_filt,real_filt)
    s_index_pred, s_pred = sortlist_by_index(id_pred_filt,pred_filt)
    return s_index_real,s_real,s_index_pred,s_pred

def distancia_textual(real_list, pred_list, conf_cut = 0.9):
    result = []
    validation = []
    if len(pred_list)==0 or len(real_list)==0:
        result = [0,0,0,0,0,0,0,0]
        validation = [0,"Reprovado"]
    else:        
        for real, pred in zip(real_list, pred_list):
            if type(real)==str and type(pred)==str:
                norm_dist_temp = textdistance.damerau_levenshtein.normalized_similarity(real, pred)
                if norm_dist_temp < conf_cut:
                    dist = textdistance.damerau_levenshtein.distance(real, pred)
                    if dist == 1:
                        norm_dist_temp = conf_cut                        
            else:
                norm_dist_temp = 0
            result.append( round(norm_dist_temp,2) )
            
        norm_dist_media = min(result)      
        validation.append( round(norm_dist_media,2) )
        if norm_dist_media >= conf_cut:
            validation.append("Aprovado")
        else:
            validation.append("Reprovado")
    f = [result, validation]
    return f

def similarity_validation( id_real, info_real, id_predito, info_predito, nivel_conf = 0.9 ):
    ## Tratando informacao predita pelo modelo:
    info_predito_trat = ocr_result_treatment(info_predito)
        
    ## Pareando as duas listas para ficar apenas a interseccao:
    id_real_f, info_real_f, id_predito_f, info_predito_f = matching_lists( id_real, info_real, id_predito, info_predito_trat  )
    
    ## Calculando medidas de similaridade e validando
    r1 = []
    r2 = []
    for xx, yy in zip(info_real_f,info_predito_f):
        r1a , r2a = distancia_textual(xx,yy, conf_cut = nivel_conf)
        r1.append(r1a)
        r2.append(r2a)
    
    ## Criando uma nova lista do tamanho da lista predita com resultados
    r1_final = []
    r2_final = []
    for j in id_predito:
        flag = np.where( np.isin( id_predito_f , j  ))[0].tolist()
        if len(flag) == 0:
            r1_final.append([])
            r2_final.append([])
        else:
            r1_temp = r1[flag[0]]
            r2_temp = r2[flag[0]]
            r1_final.append( r1_temp )
            r2_final.append(r2_temp)
    
    return r2_final, r1_final    