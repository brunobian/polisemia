from transformers import LlamaTokenizerFast,AutoModelForCausalLM,GPT2Tokenizer  
from torch.nn import functional as f 
import torch
import pandas as pd
from tqdm import tqdm

m = "GPT2"#
m = "Llama2"#

if m == "GPT2":
    ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/clm-spanish"
    model     = AutoModelForCausalLM.from_pretrained(ruta,device_map="auto")                  
    tokenizer = GPT2Tokenizer.from_pretrained(ruta)
elif m== "Llama2":
    ruta = "/data/brunobian/Documents/Repos/Repos_Analisis/awdlstm-cloze-task/data/models/Llama-2-7b-hf/snapshots/3f025b66e4b78e01b4923d510818c8fe735f6f54"
    #model = AutoModelForCausalLM.from_pretrained(ruta, device_map="auto",load_in_8bit=True)    
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto",load_in_8bit=True,cache_dir="/data/brunobian/languageModels/")    
    old_tokenizer = LlamaTokenizerFast.from_pretrained(ruta)                            
else:
    print("modelo incorrecto")

from datasets import load_dataset
raw_datasets = load_dataset("wikipedia", language="es", date="20240401")


raw_datasets = load_dataset("code_search_net", "python")

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)


'''
ids_tokens = tokenizer.encode("mi casamiento", return_tensors="pt").to('cuda') 
print(ids_tokens)
for i in (ids_tokens[0]):
    print(tokenizer.decode(i))

'''
pass