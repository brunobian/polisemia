import pandas as pd
from runModel import *

'''
DF RETURNED USING ONLY STIMULI OR IF STIMULI IS MERGED TO RESULTS:
rid   wordID      target                oracion  meaningID      significado  contextID                                           Contexto combinationID
  0        0        raya  Daniel observó la ...          1    [(Animales,1]          1  El mar, vasto e inexplorado, es hogar de una a...           1100
  1        0        raya  Daniel observó la ...          1   [(Animales,1)]          2  Las líneas son elementos básicos en el diseño ...           2100
  3        0        raya  Daniel observó la ...          2  [(Geometría,1)]          1  El mar, vasto e inexplorado, es hogar de una a...           1200
  5        0        raya  Daniel observó la ...          2  [(Geometría,1)]          3  La tilde diacrítica es la que permite distingu...           3200
'''

def get_significados_sin_peso(row):
    return [x[0] for x in row.significado]

def get_pesos(row):
    return [x[1] for x in row.significado]

## Para obtener los sesgos con el modelo una vez por fila
def get_diference_multiple_context_all_layer(listaSignificados, pesoDeSignificados, contexto, target, oracion, layers, m, model, tokenizer):
    sesgos_por_fila = []
    #pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "." # TODO: Ver cuando se usaria esto
    query = get_query(contexto, oracion)
    text_ids = tokenize(query, tokenizer, m)
    sig_ids, ids_target_in_text, target_token = find_target(text_ids, target, listaSignificados, m, tokenizer) # TODO: No esta en uso find_significado, si lo saco, eliminar lo que devuelvo y no se usa
    #find_significado(text_ids, text_ids[0].tolist(), sig_ids.tolist(), tokenizer) # No esta en uso find_significado, estaba comentado, ver si lo dejo o lo saco
    hidden_state = instance_model(text_ids, model) 
    sig_embeddings = get_sig_embedding(model, m, sig_ids, pesoDeSignificados)
    for layer in layers:
        if layer == 0:
            target_embeddings = get_embedding_before_model(model, m, target_token)
            sesgos_por_fila.append(get_diference_target_sig(target_embeddings, sig_embeddings))
        else:
            target_embeddings = extract_embedding_from_layer(hidden_state, ids_target_in_text, layer)
            sesgos_por_fila.append(get_diference_target_sig(target_embeddings, sig_embeddings))
    ## En este punto tengo, por cada target-contexto: [sesgo_layer_0, sesgo_layer_1, ..., sesgo_layer_12]
    return sesgos_por_fila ## En este punto tengo, solo una combinacion sesgos_target-context

def get_sesgo_de_todas_capas_por_fila(row, layers, m, model, tokenizer):
    sig = get_significados_sin_peso(row)
    pesos = get_pesos(row)
    list_bias_by_layer = get_diference_multiple_context_all_layer(sig, pesos, row.Contexto, row.target, row.oracion, layers, m, model, tokenizer)
    return [row.combinationID, list_bias_by_layer]

def reordeno_matriz(sesgos, df_original):
    '''
    tengo
    [[combinationId1,   [capa0, ..., capa12]],
     ...
     combinationIdN*4, [capa0, ..., capa12]]
    '''
    titulos =["wordID", "target", "meaningID", "sesgoBase", "sesgoGen"]
    lista_plana = [elem for sublist in sesgos for elem in sublist]
    bloques = [lista_plana[i:i + 8] for i in range(0, len(lista_plana), 8)]
    pd_by_layer = []
    for nlayer in range(13):
        df=[]
        # Mostrar los resultados
        for CID1, sesgoCID1, CID2, sesgoCID2, CID3, sesgoCID3, CID4, sesgoCID4 in bloques:
            wordID = CID1[-2:]
            assert(CID1 == '30'+wordID)
            assert(CID2 == '10'+wordID)
            assert(CID3 == '31'+wordID)
            assert(CID4 == '21'+wordID)
            target = df_original[df_original['wordID']==int(wordID)]['target'].to_list()[0]
            df.append([wordID, target, 0, sesgoCID1[nlayer], sesgoCID2[nlayer]])
            df.append([wordID, target, 1, sesgoCID3[nlayer], sesgoCID4[nlayer]])
        '''
        tengo
        [capa0:  [[0, 0, sesgoBase, sesgoGen],
                  [0, 1, sesgoBase, sesgoGen]],
        ...
        capa12: [[N, 0, sesgoBase, sesgoGen],
                 [N, 1, sesgoBase, sesgoGen]]]] 
        '''
        df=pd.DataFrame(df, columns=titulos)
        df.to_csv(f"nuevo/sesgos_por_layer_{nlayer}.csv")
        pd_by_layer.append(df)
    return pd_by_layer

def getBias_forPreparedDf(df, layers, m, model, tokenizer):
    ## Para obtener los sesgos ejecutando el modelo la menor cantidad de veces
    sesgos_de_capas = (df.apply(lambda r: get_sesgo_de_todas_capas_por_fila(r, layers, m, model, tokenizer), axis=1))
    lista_de_df = reordeno_matriz(sesgos_de_capas, df)
    return lista_de_df

'''
Antes devolvia:   fila, ncontexto, sesgoBase, sesgoGen (por cada capa)
Ahora devuelvo: wordID, meaningID, sesgoBase, sesgoGen (por cada capa)
'''