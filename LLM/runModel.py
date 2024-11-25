import torch
from torch.nn import functional as f 

## Para la capa 0
def get_embedding_before_model(model, model_type, target):
    if model_type == "GPT2":
        target_embedding = model.transformer.wte(target[0])
    elif model_type == "Llama2":
        target_embedding = model.get_input_embeddings().weight[target]
    elif model_type == "GPT2_wordlevel":
        target_embedding = model.transformer.wte(target[0])
    
    stacked_target_embeddings = torch.stack([target_embedding])
    target_embeddings = torch.nanmean(stacked_target_embeddings, dim=0)

    return target_embeddings

## Para el resto de las capas
def get_query(context, oracion):
    query = context + " " + oracion
    return query

def tokenize(text, tokenizer, model_type):
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
        target_token  = tokenizer.encode(target, return_tensors="pt").to("mps")#.to('cuda')
            
    ## Obtengo los significados de la lista tokenizados en formato tensor
    list_sig_ids = []
    for sig in lista_sig:
        if model_type == "GPT2":
            #Se le pasa la palabra significado y se busca obtener una lista con los ids de los tokens contenidos en dicha palabra
            #Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
            #Tomo el primer elemento del tensor porque los ids estan dentro de una matriz de tamaño [1, n] con n cantidad de tokens
            ids = (tokenizer.encode(" " + sig, return_tensors="pt").to("mps"))#.to('cuda') 
            assert(ids.size()[0] == 1)
            sig_ids  = ids[0]
        elif model_type== "Llama2":
            sig_ids  = tokenizer.encode(sig, return_tensors="pt").to("mps")#.to('cuda')
            sig_ids  = sig_ids[0,1:]  # Saco el primer elemento que es <s>
            sig_ids_list= sig_ids.tolist() 
        elif model_type == "GPT2_wordlevel":
            sig_ids  = tokenizer.encode(sig, return_tensors='pt').to("mps")#.to('cuda')
            assert(sig_ids.size()[0] == 1)
            sig_ids  = sig_ids[0]
            #sig_ids_list= sig_ids.tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
        list_sig_ids.append(sig_ids)        
        
    ## Busco las posiciones del target en el texto
    text_ids_list = text_ids[0].tolist()
    for i in range(len(text_ids_list)):
        if targ_tokenizado == text_ids_list[i:(i+len(targ_tokenizado))]:
            break
    ids_target_in_text = list(range(i,i+len(targ_tokenizado)))
    #print(tokenizer.decode(text_ids[ids_target_in_text]))
    return list_sig_ids, ids_target_in_text, target_token

def find_significado(text_ids, text_ids_list, sig_ids_list, tokenizer):
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

def get_sig_embedding(model, model_type, sig, pesoDeSignificados):
    ## Version apta para listas de significados (donde tomo el promedio de los embeddings)
    sig_embeddings_list = []

    if model_type == "GPT2":
        for s in sig:
            # Obtengo un tensor con los embeddings de cada token que conforma la palabra significado
            sig_tokens_embeddings = model.transformer.wte(s)
            assert(sig_tokens_embeddings.size(0) >= 1)
            # Tomo el promedio de los embeddings de los tokens y luego lo formateo para que quede con shape [1,768]
            sig_tokens_embeddings = torch.mean(sig_tokens_embeddings,dim=0).unsqueeze(0)
            # Si en vez de usar el promedio, tomo el primer token (es esta opcion o el promedio)
            sig_embeddings_list.append(sig_tokens_embeddings) 
    elif model_type == "Llama2":
        for s in sig:
            sig_embeddings_list.append(model.get_input_embeddings().weight[s]) #Aca no tome el primero porque no estaba en el codigo original
    elif model_type == "GPT2_wordlevel":
        for s in sig:
            # Obtengo un tensor con los embeddings de cada token que conforma la palabra significado
            sig_tokens_embeddings = model.transformer.wte(s)
            assert(sig_tokens_embeddings.size(0) == 1)
            # Tomo el promedio de los embeddings de los tokens y luego lo formateo para que quede con shape [1,768]
            ####sig_tokens_embeddings = torch.mean(sig_tokens_embeddings,dim=0).unsqueeze(0)
            # Si en vez de usar el promedio, tomo el primer token (es esta opcion o el promedio)
            #sig_tokens_embeddings = sig_tokens_embeddings[0].unsqueeze(0)
            sig_embeddings_list.append(sig_tokens_embeddings) 

    # Concateno los embeddings de los significados en una lista para poder operar con ellos
    stacked_embeddings = torch.stack(sig_embeddings_list)
    # Creo un tensor con los pesos para poder operar con ellos
    pesoDeSignificados_tensor = torch.tensor(pesoDeSignificados).to("mps")#.to('cuda')
    # Multiplico cada embedding por el peso respectivo 
    stacked_embeddings_pesados = stacked_embeddings * pesoDeSignificados_tensor.unsqueeze(1)
    # Los sumo para poder hacer luego el promedio ponderado
    sig_embeddings_sum = torch.sum(stacked_embeddings_pesados, dim=0)
    # Hacer la division por la suma de los pesos (promedio ponderado)
    sig_embeddings_prom = torch.div(sig_embeddings_sum, sum(pesoDeSignificados))
    
    # VERSION PREVIA (sin lista de significados)
    #stacked_embeddings = torch.stack(sig_embeddings_list)
    #sig_embeddings = torch.nanmean(stacked_embeddings, dim=0)

    # Corro el modelo para el significado
    # output_sig = model(sig_ids, output_hidden_states=True)            
    # last_layer  = output_sig.hidden_states[-2:][0].nanmean(0) 
    # sig_embeddings = last_layer[1:]

    return sig_embeddings_prom

def get_diference_target_sig(target_embeddings, sig_embeddings):
    ## Hacemos promedios por la primera dimension en caso de tokens subpalabras
    target_av_embedding = target_embeddings.nanmean(0)
    sig_av_embedding    = sig_embeddings.nanmean(0)
    dist = f.cosine_similarity(target_av_embedding,sig_av_embedding,0).item()

    #dist = max([f.cosine_similarity(target_embeddings[i,:],sig_embeddings[j,:],0).item()
    #     for i in range(target_embeddings.size(0)) 
    #            for j in range(sig_embeddings.size(0))])
    return dist