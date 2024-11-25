"""
Scrip para analizar las palabras. Devuelve dos cosas:

*Palabras que tienen más de un token con tokenizador original
*Palabras que tienen el unk_token con el nuevo tokenizador
"""
import pandas as pd
from transformers import AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
from tqdm import tqdm
import pickle
import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing
import torch.nn.functional as F
from cambio_tokenizador.custom_tokenizer import CustomTokenizer


#Cargar tokenizadores
#base: sin fintunning
#wl: finetunnig Word Level
unk_token = '<unk>'
method = 'stimuli'

ruta_base = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
base_model = AutoModelForCausalLM.from_pretrained(ruta_base,device_map="auto")                  
tokenizer_base = GPT2Tokenizer.from_pretrained(ruta_base)

ruta_tokenizer_wl = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/tokenizer'
ruta_wl = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/model/checkpoint-21940'
wl_model = AutoModelForCausalLM.from_pretrained(ruta_wl, device_map = "auto") 
tokenizer_wl = CustomTokenizer.from_pretrained(ruta_tokenizer_wl)

if method == 'stimuli':
    df = pd.read_csv('Stimuli.csv', sep = ',')
    palabras = df['target']
elif method == 'wikitext':
    with open('tokenizer/vocab.json', 'r') as file:
        vocab = json.load(file)  
    palabras = list(vocab.keys())

#Creo listas para guardar las palabras  
mas_un_token = []
is_unk_token = []

base_embeddings, wl_embeddings = [], []
if method == 'stimuli':
    for palabra in palabras:
        #Tokenizo la palabra con ambos tokenizadores    
        base_tokens = tokenizer_base.tokenize(palabra) 
        base_tokens_tensor = tokenizer_base.encode(palabra, return_tensors = 'pt').to('cuda')

        wl_encoded = tokenizer_wl.encode(palabra) 
        wl_tokens = wl_encoded[0]

        #Calculo Embeddings
        base_outputs = base_model.transformer(torch.tensor(base_tokens_tensor)) 
        base_emb = base_outputs.last_hidden_state.mean(dim = 1).squeeze().detach()
        #base_outputs.last_hidden_state tiene shape (batch, SL, HD)

        wl_ouputs = wl_model.transformer(torch.tensor(wl_encoded).to('cuda'))
        wl_emb = wl_ouputs.last_hidden_state.detach()

        base_embeddings.append(base_emb)
        wl_embeddings.append(wl_emb)

        if wl_tokens == unk_token:
            is_unk_token.append(palabra) 

        if len(base_tokens) > 1:
            mas_un_token.append(palabra)

    print("Palabras con más de un token")
    print(len( mas_un_token ))    

    print("Palabras con unk token")
    print(len( is_unk_token ))

    distances = []
    #Computar distancias coseno
    for emb1, emb2 in zip(base_embeddings, wl_embeddings):
        dist = F.cosine_similarity(emb1, emb2)
        distances.append(1 - dist)

    #df_out = pd.DataFrame(
    #    {'Palabra': palabras,
    #     'Distancia Embeddings': distances}
    #)
    #df_out.to_csv('new_model_distances.csv')
elif method == 'wikitext':
    for palabra in palabras:
        #Tokenizo la palabra con ambos tokenizadores    
        base_tokens = tokenizer_base.tokenize(palabra) 
        base_tokens_tensor = tokenizer_base.encode(palabra, return_tensors = 'pt').to('cuda')

        wl_encoded = tokenizer_wl.encode(palabra) 
        wl_tokens = wl_encoded[0]

        if wl_tokens == unk_token:
            is_unk_token.append(palabra) 

        if len(base_tokens) > 1:
            mas_un_token.append(palabra)

    print("Palabras con más de un token")
    print(len( mas_un_token ))    

    print("Palabras con unk token")
    print(len( is_unk_token ))

