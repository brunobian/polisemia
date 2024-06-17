from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing

'''
Observar que hay partes del codigo que estan comentadas, leer lo siguiente antes de ejecutar:
- En el metodo "cargar_modelo" revisar que el path a usar este en la maquina donde estes ejecutando
    - En la de Bruno: "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
    - En la de Belu: "/Users/NaranjaX/Desktop/tesis/clm-spanish"
- El modelo para Llama2 puede instalarse cada vez o buscarse en un repo en caso de estar instalado
- Hay que usar ".to(cuda)" si usas la compu de Bruno y ".to("mps")" sino en la mac
'''

## Para preparar el experimento
def cargar_modelo(model_type, model_path=""):
    #TODO: Revisar que seria el modal path y cuando pasarlo
    #TODO: Sacar estos if usando algun patron de diseño
    if model_type == "GPT2":
        ruta      = "/Users/NaranjaX/Desktop/tesis/clm-spanish"#"/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
        model     = AutoModelForCausalLM.from_pretrained(ruta,device_map="auto")                  
        tokenizer = GPT2Tokenizer.from_pretrained(ruta)
    elif model_type == "Llama2":
        ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/Llama-2-7b-hf/snapshots/3f025b66e4b78e01b4923d510818c8fe735f6f54"
        model = AutoModelForCausalLM.from_pretrained(ruta, device_map="auto",load_in_8bit=True)    
        #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto",load_in_8bit=True,cache_dir="/data/brunobian/languageModels/")    
        tokenizer = LlamaTokenizerFast.from_pretrained(ruta)                            
    elif model_type == "GPT2_wordlevel":
        tokenizer_dict_path = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/tokenizer/token_dict.pkl'
        ruta = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/checkpoint-29160'
        model = AutoModelForCausalLM.from_pretrained(ruta).to("mps")#.to('cuda')
        try:
            with open(tokenizer_dict_path, 'rb') as tokenizer_file:
                tokenizer_dict = pickle.load(tokenizer_file)
                word2int = tokenizer_dict['word2int']
                int2word = tokenizer_dict['int2word']
                unk_token = tokenizer_dict['UNK']
        except Exception as e:
            print(f"Failed to load tokenizer dictionary: {e}")

        tokenizer = Tokenizer(WordLevel(vocab = word2int, unk_token = unk_token))
        tokenizer.pre_tokenizer = Sequence([Punctuation('removed'), Whitespace()])
    else:
        print("modelo incorrecto")
    #TODO: Ver que nos gustaria devolver
    return model, tokenizer
    
def cargar_stimuli(stimuli_path):
    return pd.read_csv(stimuli_path,quotechar='"')

def get_iterador (row):
    sig     = [row.significado1.lower().split(","), row.significado2.lower().split(",")] 
    context = [row.Contexto3, row.Contexto1, row.Contexto2]
    iterar_sobre = [(sig[0],context[0]), (sig[0],context[1]), (sig[1],context[0]), (sig[1],context[2])]
    return iterar_sobre

## Para la capa 0
def get_embedding_before_model(model, model_type, sig, target):
    ## Version apta para listas de significados (donde tomo el promedio de los embeddings)
    sig_embeddings_list = []

    if model_type == "GPT2":
        target_embedding = model.transformer.wte(target[0])
        for s in sig:
            sig_embeddings_list.append(model.transformer.wte(s[0])) #Tomo el primer id para cada palabra de la lista de significado
    elif model_type == "Llama2":
        target_embedding = model.get_input_embeddings().weight[target]
        for s in sig:
            sig_embeddings_list.append(model.get_input_embeddings().weight[s]) #Aca no tome el primero porque no estaba en el codigo original
    elif model_type == "GPT2_wordlevel":
        target_embedding = model.transformer.wte(target[0])
        for s in sig:
            sig_embeddings_list.append(model.transformer.wte(s[0])) #Tomo el primer id para cada palabra de la lista de significado

    stacked_sig_embeddings = torch.stack(sig_embeddings_list)
    sig_embeddings = torch.nanmean(stacked_sig_embeddings, dim=0)
    stacked_target_embeddings = torch.stack([target_embedding])
    target_embeddings = torch.nanmean(stacked_target_embeddings, dim=0)

    # Corro el modelo para el significado
    # output_sig = model(sig_ids, output_hidden_states=True)            
    # last_layer  = output_sig.hidden_states[-2:][0].nanmean(0) 
    # sig_embeddings = last_layer[1:]

    return sig_embeddings, target_embeddings

## Para el resto de las capas
def get_query(context, oracion):
    query = context + " " + oracion + "."
    return query

def tokenize(text, tokenizer, model_type):
    if model_type == "GPT2_wordlevel":
        text_ids = torch.tensor([tokenizer.encode(text).ids]).to("mps")#.to('cuda')
    else:
        text_ids = tokenizer.encode(text, return_tensors="pt").to("mps")#.to('cuda') 
    return text_ids

def find_target(text_ids, target, lista_sig, model_type, tokenizer):
    ## Obtengo el target tokenizado y en formato tensor
    if model_type == "GPT2":
        targ_tokenizado = tokenizer.encode(" " + target) # GPT
        target_token  = tokenizer.encode(" " + target, return_tensors="pt").to("mps")#.to('cuda') 
    elif model_type== "Llama2":
        targ_tokenizado = tokenizer.encode(target)[1:] # Llama
        target_token  = tokenizer.encode(target, return_tensors="pt").to("mps")#.to('cuda')
    elif model_type == "GPT2_wordlevel":
        targ_tokenizado = tokenizer.encode(target) # GPT
        target_token  = torch.tensor([tokenizer.encode(target).ids]).to("mps")#.to('cuda')
            
    ## Obtengo los significados de la lista tokenizados en formato tensor
    list_sig_ids = []
    for sig in lista_sig:
        if model_type == "GPT2":
            # TODO: leer como funciona el encode si sig fuese una lista
            sig_ids  = tokenizer.encode(" " + sig, return_tensors="pt").to("mps")#.to('cuda') 
            sig_ids_list= sig_ids[0].tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
        elif model_type== "Llama2":
            sig_ids  = tokenizer.encode(sig, return_tensors="pt").to("mps")#.to('cuda')
            sig_ids  = sig_ids[0,1:]  # Saco el primer elemento que es <s>
            sig_ids_list= sig_ids.tolist() 
        elif model_type == "GPT2_wordlevel":
            sig_ids  = torch.tensor([tokenizer.encode(sig).ids]).to("mps")#.to('cuda')
            sig_ids_list= sig_ids.tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
        list_sig_ids.append(sig_ids)        
        
    ## Busco las posiciones del target en el texto
    text_ids_list = text_ids[0].tolist()
    for i in range(len(text_ids_list)):
        if targ_tokenizado == text_ids_list[i:(i+len(targ_tokenizado))]:
            break
    ids_target_in_text = list(range(i,i+len(targ_tokenizado)))
    #print(tokenizer.decode(text_ids[ids_target_in_text]))
    return list_sig_ids, ids_target_in_text, target_token

def find_significado(text_ids, text_ids_list, sig_ids_list):
    #Busco las posiciones del significado en el texto
    for i in range(len(text_ids_list)):                                 
        if sig_ids_list == text_ids_list[i:(i+len(sig_ids_list))]:              
            break                                                       
    ids_sig_in_text = list(range(i,i+len(sig_ids_list))) 
    print(tokenizer.decode(text_ids[ids_sig_in_text]))  

def instance_model(text_ids, model):
    # Corro el modelo para el texto 
    # y extraigo embedding del target contextualizado
    output_text = model(text_ids, output_hidden_states=True)
    h = output_text.hidden_states 
    return h

def extract_embedding_from_layer(hidden_state, ids_target_in_text, layer=-1):
    last_layer  = hidden_state[layer][0]
    target_embeddings = last_layer[ids_target_in_text]
    #sig_embeddings    = last_layer[ids_sig_in_text]
    return target_embeddings

def get_sig_embedding(model, model_type, sig):
    ## Version apta para listas de significados (donde tomo el promedio de los embeddings)
    sig_embeddings_list = []

    if model_type == "GPT2":
        for s in sig:
            sig_embeddings_list.append(model.transformer.wte(s[0])) #Tomo el primer id para cada palabra de la lista de significado
    elif model_type == "Llama2":
        for s in sig:
            sig_embeddings_list.append(model.get_input_embeddings().weight[s]) #Aca no tome el primero porque no estaba en el codigo original
    elif model_type == "GPT2_wordlevel":
        for s in sig:
            sig_embeddings_list.append(model.transformer.wte(s[0])) #Tomo el primer id para cada palabra de la lista de significado

    stacked_embeddings = torch.stack(sig_embeddings_list)
    sig_embeddings = torch.nanmean(stacked_embeddings, dim=0)

    # Corro el modelo para el significado
    # output_sig = model(sig_ids, output_hidden_states=True)            
    # last_layer  = output_sig.hidden_states[-2:][0].nanmean(0) 
    # sig_embeddings = last_layer[1:]

    return sig_embeddings

def get_diference_target_sig(target_embeddings, sig_embeddings):
    target_av_embedding = target_embeddings.nanmean(0)
    sig_av_embedding    = sig_embeddings.nanmean(0)
    dist = f.cosine_similarity(target_av_embedding,sig_av_embedding,0).item()

    #dist = max([f.cosine_similarity(target_embeddings[i,:],sig_embeddings[j,:],0).item()
    #     for i in range(target_embeddings.size(0)) 
    #            for j in range(sig_embeddings.size(0))])
    return dist

## Para obtener los sesgos con el modelo una vez por capa
def get_diference_multiple_context(list_sig_y_contexto, target, oracion, layer):
    sesgos_en_capa_dada = []
    for s, c in list_sig_y_contexto:
        #pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "." # TODO: Ver cuando se usaria esto
        query = get_query(c, oracion)
        text_ids = tokenize(query, tokenizer, m)
        sig_ids, ids_target_in_text, target_token = find_target(text_ids, target, s, m, tokenizer) # TODO: No esta en uso find_significado, si lo saco, eliminar lo que devuelvo y no se usa
        #find_significado(text_ids, text_ids[0].tolist(), sig_ids.tolist()) # No esta en uso find_significado, estaba comentado, ver si lo dejo o lo saco
        if layer == 0 :
            sig_embeddings, target_embeddings = get_embedding_before_model(model, m, sig_ids, target_token)
        else :
            hidden_state = instance_model(text_ids, model) 
            sig_embeddings = get_sig_embedding(model, m, sig_ids)
            #TODO: Pensar si hay una mejor manera de guardarme los valores para aprovechar el armado del modelo solo una vez
            '''dist = []
            for layer in layers:
                target_embeddings = extract_embedding_from_layer(hidden_state, ids_target_in_text, layer)
                dist.append(get_diference_target_sig(target_embeddings, sig_first_embeddings))
            promedio_dist = sum(dist)/len(dist)
            sesgo.append(promedio_dist)'''
            target_embeddings = extract_embedding_from_layer(hidden_state, ids_target_in_text, layer)
        sesgo_en_capa = get_diference_target_sig(target_embeddings, sig_embeddings)
        sesgos_en_capa_dada.append(sesgo_en_capa)
    return sesgos_en_capa_dada

def get_sesgo_por_fila(row, layer):
    return get_diference_multiple_context(get_iterador(row), row.target, row.oracion, layer)

def get_sesgo_por_capas(df, layers):
    ## Para graficar una linea con errores estandar
    error_promedio_por_capa = []
    error_estandar_por_capa = []
    ## Para graficar con boxplot
    errores_por_capa = []
    for layer in layers:
        all_sesgos_layer= (df.apply(lambda r: get_sesgo_por_fila(r, layer), axis=1))
        df_por_layer = get_df_de_sesgo_del_modelo(all_sesgos_layer, layer)
        error_promedio, error_estandar, errores = calculo_error(df_por_layer)
        ## Para graficar una linea con errores estandar
        error_promedio_por_capa.append(error_promedio)
        error_estandar_por_capa.append(error_estandar)
        ## Para graficar con boxplot
        errores_por_capa.append(errores)
    return errores_por_capa, error_promedio_por_capa, error_estandar_por_capa

## Para obtener los sesgos con el modelo una vez por fila
def get_diference_multiple_context_all_layer(list_sig_y_contexto, target, oracion, layers):
    sesgos_en_capa_dada = []
    for s, c in list_sig_y_contexto:
        sesgos_por_fila = []
        #pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "." # TODO: Ver cuando se usaria esto
        query = get_query(c, oracion)
        text_ids = tokenize(query, tokenizer, m)
        sig_ids, ids_target_in_text, target_token = find_target(text_ids, target, s, m, tokenizer) # TODO: No esta en uso find_significado, si lo saco, eliminar lo que devuelvo y no se usa
        #find_significado(text_ids, text_ids[0].tolist(), sig_ids.tolist()) # No esta en uso find_significado, estaba comentado, ver si lo dejo o lo saco
        hidden_state = instance_model(text_ids, model) 
        sig_embeddings_after_model = get_sig_embedding(model, m, sig_ids)
        for layer in layers:
            if layer == 0:
                sig_embeddings, target_embeddings = get_embedding_before_model(model, m, sig_ids, target_token)
                sesgos_por_fila.append(get_diference_target_sig(target_embeddings, sig_embeddings))
            else:
                target_embeddings = extract_embedding_from_layer(hidden_state, ids_target_in_text, layer)
                sesgos_por_fila.append(get_diference_target_sig(target_embeddings, sig_embeddings_after_model))
        sesgos_en_capa_dada.append(sesgos_por_fila)
        ## En este punto tengo, por cada target-contexto: [sesgo_layer_0, sesgo_layer_1, ..., sesgo_layer_12]
    return sesgos_en_capa_dada ## En este punto tengo, [sesgos_target-context_1, sesgos_target-context_2, sesgos_target-context_3, sesgos_target-context_4]

def get_sesgo_de_todas_capas_por_fila(row, layers):
    return get_diference_multiple_context_all_layer(get_iterador(row), row.target, row.oracion, layers)

def reordeno_sesgos(sesgos):
    '''
    tenemos:
        [fila 1: [Target-Contexto 1: [sc1tc1f1, sc2tc1f1, ... , sc12tc1f1],      
                  Target-Contexto 2: [sc1tc2f1, sc2tc2f1, ... , sc12tc2f1],  
                  Target-Contexto 3: [sc1tc3f1, sc2tc3f1, ... , sc12tc3f1],  
                  Target-Contexto 4: [sc1tc4f1, sc2tc4f1, ... , sc12tc4f1]], 
        ...
        fila N:  [Target-Contexto 1: [sc1tc1fN, sc2tc1fN, ... , sc12tc1fN],      
                  Target-Contexto 2: [sc1tc2fN, sc2tc2fN, ... , sc12tc2fN],  
                  Target-Contexto 3: [sc1tc3fN, sc2tc3fN, ... , sc12tc3fN],  
                  Target-Contexto 4: [sc1tc4fN, sc2tc4fN, ... , sc12tc4fN]]]
    queremos:
        [sesgos_capa_1: [fila_1: [sc1tc1f1, sc1tc2f1, sc1tc3f1, sc1tc4f1], 
                         ...
                         fila_N: [sc1tc1fN, sc1tc2fN, sc1tc3fN, sc1tc4fN]],
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

def get_sesgo_para_todas_las_capas(df, layers):
    ## La matriz va a tener en cada fila los sesgos de las capas indicadas
    sesgos_de_capas_por_fila = (df.apply(lambda r: get_sesgo_de_todas_capas_por_fila(r, layers), axis=1))
    sesgos_por_capa = reordeno_sesgos(sesgos_de_capas_por_fila)
    ## Para graficar una linea con errores estandar
    error_promedio_por_capa = []
    error_estandar_por_capa = []
    ## Para graficar con boxplot
    errores_por_capa = []
    for layer in layers:
        df_por_layer = get_df_de_sesgo_del_modelo(sesgos_por_capa[layer], layer)
        error_promedio, error_estandar, errores = calculo_error(df_por_layer)
        ## Para graficar una linea con errores estandar
        error_promedio_por_capa.append(error_promedio)
        error_estandar_por_capa.append(error_estandar)
        ## Para graficar con boxplot
        errores_por_capa.append(errores)
    return errores_por_capa, error_promedio_por_capa, error_estandar_por_capa

## Para las metricas
def calculo_distancia_entre_sesgoBase_sesgoGenerado(row):
    ## Obtengo la distancia ortogonal a la identidad
    #TODO: sumar el valor absoluto del minimo
    e = 1
    return (row.sesgoGen - row.sesgoBase)

def calculo_error(df):
    df_contexto_1 = df.get(['numContexto1', 'sesgoBase1', 'sesgoGen1'])
    df_contexto_1 = df_contexto_1.set_axis(["numContexto", "sesgoBase", "sesgoGen"], axis=1)
    df_contexto_2 = df.get(['numContexto2', 'sesgoBase2', 'sesgoGen2'])
    df_contexto_2 = df_contexto_2.set_axis(["numContexto", "sesgoBase", "sesgoGen"], axis=1)
    df_por_contexto = pd.concat([df_contexto_1, df_contexto_2])
    df_por_contexto.to_csv(f"distancias_por_layer.csv")
    ## Calculo el promedio de las distancias ortogonales a la identidad contando a cada target con un contexto por separado
    ## Entonces debo sumar las distancias de un mismo target pero distintos contextos
    ## La cantidad de distancias va a ser el doble de la cantidad de targets porque para cada uno hay dos contextos
    distancias_por_contexto = df_por_contexto.apply(lambda x : calculo_distancia_entre_sesgoBase_sesgoGenerado(x), axis = 1)
    error_promedio = distancias_por_contexto.mean()
    error_estandar = distancias_por_contexto.sem()
    return error_promedio, error_estandar, distancias_por_contexto.to_list()

def get_df_de_sesgo_del_modelo(all_sesgos, name=''):
    df=[]
    titulos =["fila", "numContexto1", "sesgoBase1", "sesgoGen1", "numContexto2", "sesgoBase2", "sesgoGen2" ]
    for i,p in enumerate(all_sesgos):
        df.append([i, 1, p[0], p[1], 2, p[2], p[3]])

    df=pd.DataFrame(df, columns=titulos)
    #df.to_csv(f"distancias_por_layer_{name}.csv")
    return df

def get_plot(distances, errores_estandar, layers):
    '''figsize=(10,10))'''
    plt.errorbar(layers, distances, yerr=(errores_estandar))
    plt.title("Sesgos segun cada capa de GPT2")
    plt.xlabel("Capas")
    plt.xticks(layers)
    plt.ylabel("Error promedio")
    plt.yticks(distances)
    plt.savefig('pltLayers_-layers.png')
    plt.show()

def get_plot_with_boxplot(distances, errores_promedio, errores_estandar, layers):
    plt.figure(figsize=(10,15))
    plt.boxplot(distances)
    plt.errorbar(layers, errores_promedio, yerr=(errores_estandar))
    plt.title("Sesgos segun cada capa de GPT2")
    plt.xlabel("Capas")
    plt.xticks(layers)
    plt.ylabel("Error promedio")
    #plt.yticks(errores_promedio)
    plt.savefig('pltLayers_-layers.png')
    plt.show()

m = "GPT2"#"Llama2"#
model, tokenizer = cargar_modelo(m) #TODO: Pensar mejor como devolver esto, si hace falta estar pasando las tres cosas o que
df = cargar_stimuli("Stimuli.csv") #"Stimuli.csv"
layers = [0,1,2,3,4,5,6,7,8,9,10,11,12]
'''
all_sesgos = []
all_sesgos = (df.apply(lambda r: get_sesgo_por_fila(r, layers), axis=1))
get_df_de_sesgo_del_modelo(all_sesgos)
'''
## Para obtener los sesgos ejecutando el modelo la menor cantidad de veces
errores_por_capa, error_promedio_por_capa, error_estandar_por_capa = get_sesgo_para_todas_las_capas(df, layers)

##Para obtener sesgos ejecutando todas las veces
#errores_por_capa, error_promedio_por_capa, error_estandar_por_capa = get_sesgo_por_capas(df, layers)

## Para graficar una linea con errores estandar
#get_plot(error_promedio_por_capa, error_estandar_por_capa, layers)
## Para graficar con boxplot
get_plot_with_boxplot(errores_por_capa, error_promedio_por_capa, error_estandar_por_capa, layers)

'''
OK Cambiar el nombre del eje y "Error promedio"
OK Agregar la capa 0 la distancia entre embeddings estaticos y el significado
OKHacer esto mismo para la target: get_sig_embedding
OK hacer reshape del df con los resultados para tener todo en una misma columna
OK Ademas del promedio(df.mean(columna)) calcular el error estandar (df.sem(columna))
OK Usar:
    plt.errorbar(capas, promedio de capa, yerr=(error estandar), fmt="o")
    plt.show()

buscar una palabra que se sample mal, re bien y promedio
analizar misma palabra distintos contextos
OK agregar en la resta la division por sesgo base para normalizar
Sumamos uno a cada sesgo porque la distancia coseno va de -1 a 1 y queremos evitar los valores negativos

'''