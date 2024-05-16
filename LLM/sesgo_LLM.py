"""from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
from tqdm import tqdm
import pickle

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing



m = "GPT2_wordlevel"#"Llama2"#

if m == "GPT2":
    ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
    model     = AutoModelForCausalLM.from_pretrained(ruta,device_map="auto")                  
    tokenizer = GPT2Tokenizer.from_pretrained(ruta)
elif m== "Llama2":
    ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/Llama-2-7b-hf/snapshots/3f025b66e4b78e01b4923d510818c8fe735f6f54"
    model = AutoModelForCausalLM.from_pretrained(ruta, device_map="auto",load_in_8bit=True)    
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto",load_in_8bit=True,cache_dir="/data/brunobian/languageModels/")    
    tokenizer = LlamaTokenizerFast.from_pretrained(ruta)                            
elif m == "GPT2_wordlevel":
    tokenizer_dict_path = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/tokenizer/token_dict.pkl'
    ruta = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/models/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish_word_stimulidb+reddit/checkpoint-29160'
    model = AutoModelForCausalLM.from_pretrained(ruta).to('cuda')
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
    
df=pd.read_csv("Stimuli.csv",quotechar='"')

all_sesgos = []
for iR,r in tqdm(df.iterrows()):
    
    target  = r.target
    oracion = r.oracion
    sig     = [r.significado1.lower().split(","), r.significado2.lower().split(",")] # Esto va a pasar a ser una lista de listas
    context = [r.Contexto3, r.Contexto1, r.Contexto2]

    # titulos =      ["sesgoBase1",        "sesgoGen1",         "sesgoBase2",        "sesgoGen2" ]
    # iterar_sobre = [(sig[0],context[0]), (sig[0],context[1]), (sig[1],context[0]), (sig[1],context[2])]
    # for s,c in iterar_sobre:
    #     print(s,c)

        
    sesgo = []
    for c in context:
        sims = []
        for s in sig:
            pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "."
            query = c + " " + oracion + "."
            if m == "GPT2_wordlevel":
                text_ids = torch.tensor([tokenizer.encode(query).ids]).to('cuda')
            else:
                text_ids = tokenizer.encode(query, return_tensors="pt").to('cuda') 
            
            if m == "GPT2":
                targ_ids = tokenizer.encode(" " + target) # GPT
                sig_ids  = tokenizer.encode(" " + s, return_tensors="pt").to('cuda') # ver que funciona a una lista
                sig_ids_list= sig_ids[0].tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
                
            elif m== "Llama2":
                targ_ids = tokenizer.encode(target)[1:] # Llama
                sig_ids  = tokenizer.encode(s, return_tensors="pt").to('cuda')
                sig_ids  = sig_ids[0,1:]  # Saco el primer elemento que es <s>
                sig_ids_list= sig_ids.tolist() 
            elif m == "GPT2_wordlevel":
                targ_ids = tokenizer.encode(target) # GPT
                sig_ids  = torch.tensor([tokenizer.encode(s).ids]).to('cuda')
                sig_ids_list= sig_ids.tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
 
            # Busco las posiciones del target en el texto
            text_ids_list = text_ids[0].tolist()
            for i in range(len(text_ids_list)):
                if targ_ids == text_ids_list[i:(i+len(targ_ids))]:
                    break
            ids_target_in_text = list(range(i,i+len(targ_ids)))
            #print(tokenizer.decode(text_ids[ids_target_in_text]))
            
            # Busco las posiciones del significado en el texto
            #for i in range(len(text_ids_list)):                                 
            #    if sig_ids_list == text_ids_list[i:(i+len(sig_ids_list))]:              
            #        break                                                       
            #ids_sig_in_text = list(range(i,i+len(sig_ids_list))) 
            #print(tokenizer.decode(text_ids[ids_sig_in_text]))  

            # Corro el modelo para el texto 
            # y extraigo embedding del target contextualziado
            output_text = model(text_ids, output_hidden_states=True)
            h = output_text.hidden_states
            last_layer  = h[-5:][0].nanmean(0)
            #last_layer = torch.cat((h[-4],h[-3],h[-2],h[-1]),dim=2)[0]
            target_embeddings = last_layer[ids_target_in_text]
            #sig_embeddings    = last_layer[ids_sig_in_text]

            if m == "GPT2":
                sig_embeddings = model.transformer.wte(sig_ids[0])
            elif m == "Llama2":
                sig_embeddings = model.get_input_embeddings().weight[sig_ids]
            elif m == "GPT2_wordlevel":
                sig_embeddings = model.transformer.wte(sig_ids[0])


            # Corro el modelo para el significado
            # output_sig = model(sig_ids, output_hidden_states=True)            
            # last_layer  = output_sig.hidden_states[-2:][0].nanmean(0) 
            # sig_embeddings = last_layer[1:]

            target_av_embedding = target_embeddings.nanmean(0)
            sig_av_embedding    = sig_embeddings.nanmean(0)
            dist = f.cosine_similarity(target_av_embedding,sig_av_embedding,0).item()

            #dist = max([f.cosine_similarity(target_embeddings[i,:],sig_embeddings[j,:],0).item()
            #     for i in range(target_embeddings.size(0)) 
            #            for j in range(sig_embeddings.size(0))])
            sims.append(dist)
        
        sesgo.append(sims)
    all_sesgos.append([sesgo[0], [sesgo[1][0], sesgo[2][1]]])

df=[]
for i,p in enumerate(all_sesgos):
    #ind = i + (i>21) + (i>3)
    df.append([i, 1, p[0][0], p[1][0], 2, p[0][1], p[1][1]])

import pandas as pd
df=pd.DataFrame(df)
#df.to_csv("distancias.csv")
df.to_csv("distancias_nuevo_modelo.csv")"""
from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
from tqdm import tqdm
import pickle

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing

#.to(cuda) va al gpu de la compu de bruno (es posible que me convenga sacarlo en la mia)

#TODO: Repensar como hacer lo de los modelos a nivel diseño
def cargar_modelo(model_type, model_path=""):
    #TODO: Revisar que seria el modal path y cuando pasarlo
    #TODO: Sacar estos if usando algun patron de diseño
    if model_type == "GPT2":
        ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
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
        model = AutoModelForCausalLM.from_pretrained(ruta).to('cuda')
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

def get_query(context, oracion):
    query = context + " " + oracion + "."
    return query

def tokenize(text, tokenizer, model_type):
    if model_type == "GPT2_wordlevel":
        text_ids = torch.tensor([tokenizer.encode(text).ids]).to('cuda')
    else:
        text_ids = tokenizer.encode(text, return_tensors="pt").to('cuda') 
    return text_ids

def find_target(text_ids, target, sig, model_type, tokenizer):
    if model_type == "GPT2":
        targ_ids = tokenizer.encode(" " + target) # GPT
        # TODO: leer como funciona el encode si sig fuese una lista
        sig_ids  = tokenizer.encode(" " + sig, return_tensors="pt").to('cuda') 
        sig_ids_list= sig_ids[0].tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
        
    elif model_type== "Llama2":
        targ_ids = tokenizer.encode(target)[1:] # Llama
        sig_ids  = tokenizer.encode(sig, return_tensors="pt").to('cuda')
        sig_ids  = sig_ids[0,1:]  # Saco el primer elemento que es <s>
        sig_ids_list= sig_ids.tolist() 

    elif model_type == "GPT2_wordlevel":
        targ_ids = tokenizer.encode(target) # GPT
        sig_ids  = torch.tensor([tokenizer.encode(sig).ids]).to('cuda')
        sig_ids_list= sig_ids.tolist()  # si la palabra del significado tiene mas de un token, me quedo con el primero
    
    # Busco las posiciones del target en el texto
    text_ids_list = text_ids[0].tolist()
    for i in range(len(text_ids_list)):
        if targ_ids == text_ids_list[i:(i+len(targ_ids))]:
            break
    ids_target_in_text = list(range(i,i+len(targ_ids)))
    #print(tokenizer.decode(text_ids[ids_target_in_text]))
    return sig_ids, text_ids_list, sig_ids_list, ids_target_in_text

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
    """ if model_type == "GPT2":
        sig_embeddings = model.transformer.wte(sig[0])
    elif model_type == "Llama2":
        sig_embeddings = model.get_input_embeddings().weight[sig]
    elif model_type == "GPT2_wordlevel":
        sig_embeddings = model.transformer.wte(sig[0]) """
    #Version apta para listas de significados (donde tomo el promedio de los embeddings)
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

def get_diference_multiple_context(list_sig_y_contexto, target, oracion, model_type, model, tokenizer, layers):
    sesgo = []
    for s,c in list_sig_y_contexto:
        pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "." # TODO: Ver cuando se usaria esto
        query = get_query(c, oracion)
        text_ids = tokenize(query, tokenizer, model_type)
        sig_ids, text_ids_list, sig_ids_list, ids_target_in_text = find_target(text_ids, target, s, model_type, tokenizer) # TODO: No esta en uso find_significado, si lo saco, eliminar lo que devuelvo y no se usa
        #find_significado(text_ids, text_ids_list, sig_ids_list) # TODO: No esta en uso find_significado, estaba comentado, ver si lo dejo o lo saco
        sig_embeddings = get_sig_embedding(model, model_type, sig_ids)
        hidden_state = instance_model(text_ids, model) 
        target_embeddings = []
        dist = []
        for layer in layers:
            target_embeddings.append(extract_embedding_from_layer(hidden_state, ids_target_in_text, layer))
            dist = get_diference_target_sig(target_embeddings, sig_embeddings)
            sesgo.append(dist)
    return sesgo

def get_sesgo_por_fila(row, layers):
    #for iR,r in tqdm(df.iterrows()): ...
    # ANTES SE HACIA ESTO CUANDO NO SE USABA ITERAR SOBRE:
    # Se creaba sesgo = [], se iteraba primero en el contexto, se creaba el sims = [] y luego se iteraba en el significado
    #    sesgo.append(sims)
    #    all_sesgos.append([sesgo[0], [sesgo[1][0], sesgo[2][1]]])
    #TODO: Revisar si se esta llenando el all_sesgos y se esta devolviendo modificado
    indTarget, _, target, _, oracion, signif1, _, contexto1, signif2, _, contexto2, _, _, contextoAmbiguo, _ = row
    sig     = [signif1.lower().split(","), signif2.lower().split(",")] 
    context = [contextoAmbiguo, contexto1, contexto2]

    #TODO: Ver en que momento pueden sernos utiles estos titulos y pasarlo a ese lugar
    titulos =      ["sesgoBase1",        "sesgoGen1",         "sesgoBase2",        "sesgoGen2" ]
    iterar_sobre = [(sig[0],context[0]), (sig[0],context[1]), (sig[1],context[0]), (sig[1],context[2])]
     
    return get_diference_multiple_context(iterar_sobre, target, oracion, m, model, tokenizer, layers)

def get_df_de_sesgo_del_modelo(all_sesgos):
    df=[]
    for i,p in enumerate(all_sesgos):
        #ind = i + (i>21) + (i>3)
        # TODO: Revisar si con el cambio de los sesgos del significado/contexto sigue ok este formato de p
        df.append([i, 1, p[0][0], p[1][0], 2, p[0][1], p[1][1]])

    df=pd.DataFrame(df)
    #df.to_csv("distancias.csv")
    df.to_csv("distancias_nuevo_modelo.csv")

m = "GPT2_wordlevel"#"Llama2"#
model, tokenizer = cargar_modelo(m) #TODO: Pensar mejor como devolver esto, si hace falta estar pasando las tres cosas o que
df = cargar_stimuli("Stimuli.csv")
layers = [-1]
all_sesgos = []
all_sesgos.append(df.apply(lambda r: get_sesgo_por_fila(r, layers), axis=1))
get_df_de_sesgo_del_modelo(all_sesgos)
