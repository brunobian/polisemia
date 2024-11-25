import pandas as pd
from runModel import *

def get_significados_sin_peso(row):
    significados = []
    signif1 = []
    for lista_signif1 in row.significado1.split(','):
        signif1.append(lista_signif1.split(';')[0].lower())
    significados.append(signif1)
    signif2 = []
    for lista_signif2 in row.significado2.split(','):
        signif2.append(lista_signif2.split(';')[0].lower())
    significados.append(signif2)
    return significados

def get_pesos(row):
    pesos = []
    pesos1 = []
    for lista_signif1 in row.significado1.split(','):
        '''SI no es entero que salte un error'''
        signif = lista_signif1.split(';')[1]
        assert(float(signif) - int(signif) == 0)
        pesos1.append(int(signif))
    pesos.append(pesos1)
    pesos2 = []
    for lista_signif2 in row.significado2.split(','):
        '''SI no es entero que salte un error'''
        signif = lista_signif2.split(';')[1]
        assert(float(signif) - int(signif) == 0)
        pesos2.append(int(signif))
    pesos.append(pesos2)
    return pesos

def get_iterador (row):
    sig = get_significados_sin_peso(row)
    pesos = get_pesos(row)
    context = [row.Contexto3, row.Contexto1, row.Contexto2]
    iterar_sobre = [(sig[0], pesos[0], context[0]), (sig[0], pesos[0], context[1]), (sig[1], pesos[1], context[0]), (sig[1], pesos[1], context[2])]
    return iterar_sobre

## Para obtener los sesgos con el modelo una vez por fila
def get_diference_multiple_context_all_layer(list_sig_y_contexto, target, oracion, layers, m, model, tokenizer):
    sesgos_en_capa_dada = []
    for listaSignificados, pesoDeSignificados, contexto in list_sig_y_contexto:
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
        sesgos_en_capa_dada.append(sesgos_por_fila)
        ## En este punto tengo, por cada target-contexto: [sesgo_layer_0, sesgo_layer_1, ..., sesgo_layer_12]
    return sesgos_en_capa_dada ## En este punto tengo, [sesgos_target-context_1, sesgos_target-context_2, sesgos_target-context_3, sesgos_target-context_4]

def get_sesgo_de_todas_capas_por_fila(row, layers, m, model, tokenizer):
    return get_diference_multiple_context_all_layer(get_iterador(row), row.target, row.oracion, layers, m, model, tokenizer)

def reordeno_matriz_sesgos(sesgos):
    '''
    tenemos:
        [fila 1: [Target-Contexto 1: [sc0tc1f1, sc1tc1f1, ... , sc12tc1f1],      
                  Target-Contexto 2: [sc0tc2f1, sc1tc2f1, ... , sc12tc2f1],  
                  Target-Contexto 3: [sc0tc3f1, sc1tc3f1, ... , sc12tc3f1],  
                  Target-Contexto 4: [sc0tc4f1, sc1tc4f1, ... , sc12tc4f1]], 
        ...
        fila N:  [Target-Contexto 1: [sc0tc1fN, sc1tc1fN, ... , sc12tc1fN],      
                  Target-Contexto 2: [sc0tc2fN, sc1tc2fN, ... , sc12tc2fN],  
                  Target-Contexto 3: [sc0tc3fN, sc1tc3fN, ... , sc12tc3fN],  
                  Target-Contexto 4: [sc0tc4fN, sc1tc4fN, ... , sc12tc4fN]]]
    queremos:
        [sesgos_capa_0: [fila_1: [sc0tc1f1, sc0tc2f1, sc0tc3f1, sc0tc4f1], 
                         ...
                         fila_N: [sc0tc1fN, sc0tc2fN, sc0tc3fN, sc0tc4fN]],
         ... , 
         sesgos_capa_12:[fila_1: [sc12tc1f1, sc12tc2f1, sc12tc3f1, sc12tc4f1], 
                         ...
                         fila_N: [sc12tc1fN, sc12tc2fN, sc12tc3fN, sc12tc4fN]],]
    '''
    sesgos_de_tc_por_capa_por_fila = []
    for sesgos_de_capa_por_targetcontext in sesgos:   
        ## Traspongo la matriz de capas por target-context a matriz de target-context por capas
        sesgos_de_targetcontext_por_capa = [[fila[i] for fila in sesgos_de_capa_por_targetcontext ] for i in range(len(sesgos_de_capa_por_targetcontext[0]))]
        sesgos_de_tc_por_capa_por_fila.append(sesgos_de_targetcontext_por_capa)
    ## Traspongo la matriz de capas por filas a matriz de filas por capas
    sesgos_por_capa = [[fila[i] for fila in sesgos_de_tc_por_capa_por_fila ] for i in range(len(sesgos_de_tc_por_capa_por_fila[0]))]
    return sesgos_por_capa

def get_lista_de_df(sesgos_por_capa):
    lista_de_df = []
    for sesgo_por_fila in sesgos_por_capa:
        lista_de_df.append(get_df_de_sesgo_del_modelo(sesgo_por_fila))
    return lista_de_df

def get_df_de_sesgo_del_modelo(all_sesgos, name=''):
    df=[]
    ## Para tener un df con una fila por palabra de target (quedan dos contextos por fila)
    #titulos =["fila", "numContexto1", "sesgoBase1", "sesgoGen1", "numContexto2", "sesgoBase2", "sesgoGen2" ]
    ## Para tener un df con una fila por palabra de target (quedan dos contextos por fila)
    titulos =["fila","numContexto", "sesgoBase", "sesgoGen"]
    for i,p in enumerate(all_sesgos):
        ## Para tener un df con una fila por palabra de target (quedan dos contextos por fila)
        #df.append([i, 1, p[0], p[1], 2, p[2], p[3]]) 
        ## Para tener un df con una fila por palabra de target+contexto
        df.append([i, 1, p[0], p[1]])
        df.append([i, 2, p[2], p[3]])

    df=pd.DataFrame(df, columns=titulos)
    #df.to_csv(f"distancias_por_layer_{name}.csv")
    return df

def getBias(df, layers, m, model, tokenizer):
    ## Para obtener los sesgos ejecutando el modelo la menor cantidad de veces
    sesgos_de_capas_por_fila = (df.apply(lambda r: get_sesgo_de_todas_capas_por_fila(r, layers, m, model, tokenizer), axis=1))
    sesgos_por_capa = reordeno_matriz_sesgos(sesgos_de_capas_por_fila)
    lista_de_df = get_lista_de_df(sesgos_por_capa)
    return lista_de_df