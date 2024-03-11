from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
from tqdm import tqdm

m = "GPT2"#"GPT2"#

if m == "GPT2":
    ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
    model     = AutoModelForCausalLM.from_pretrained(ruta,device_map="auto")                  
    tokenizer = GPT2Tokenizer.from_pretrained(ruta)
elif m== "Llama2":
    ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/Llama-2-7b-hf/snapshots/3f025b66e4b78e01b4923d510818c8fe735f6f54"
    #model = AutoModelForCausalLM.from_pretrained(ruta, device_map="auto",load_in_8bit=True)    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto",load_in_8bit=True,cache_dir="/data/brunobian/languageModels/")    
    tokenizer = LlamaTokenizerFast.from_pretrained(ruta)                            
else:
    print("modelo incorrecto")
    
df=pd.read_csv("Stimuli.csv",quotechar='"')

all_sesgos = []
for iR,r in tqdm(df.iterrows()):
    
    target  = r.target
    oracion = r.oracion
    sig     = [r.significado1.lower(), r.significado2.lower()]
    context = [r.Contexto3,    r.Contexto1, r.Contexto2]

    sesgo = []
    for c in context:
        sims = []
        for s in sig:
            pregunta = "El significado de este texto está asoacido a " + s + ""
            pregunta = "En la oración anterior el significado de la palabra " + target + " está asoacido a " + s + "."
            query = c + " " + oracion + " " + pregunta
            query = c + " " + oracion + "."
            text_ids = tokenizer.encode(query, return_tensors="pt").to('cuda') 
            
            if m == "GPT2":
                targ_ids = tokenizer.encode(" " + target) # GPT
                sig_ids  = tokenizer.encode(" " + s, return_tensors="pt").to('cuda')
                sig_ids_list= sig_ids[0].tolist() 
                
            elif m== "Llama2":
                targ_ids = tokenizer.encode(target)[1:] # Llama
                sig_ids  = tokenizer.encode(s, return_tensors="pt").to('cuda')
                sig_ids  = sig_ids[0,1:]  # Saco el primer elemento que es <s>
                sig_ids_list= sig_ids.tolist() 

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
df.to_csv("distancias.csv")
