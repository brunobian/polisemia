#!/usr/bin/env python3

"""
Change GPT-2's tokenization scheme from BPE to word-level (whitespace)
"""

import argparse
import os
import re
from pathlib import Path
import torch
from tqdm import tqdm


from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import numpy as np
from transformers import (Trainer, 
                        TrainingArguments, 
                        AutoTokenizer, 
                        AutoModelForCausalLM,
                        GPT2LMHeadModel,
                        GPT2Tokenizer,
                        PreTrainedTokenizer,
                        TrainerCallback)
from custom_tokenizer import CustomTokenizer


def split_dataset(dataset, train_ratio=0.9):
    """ Split a dataset into training and validation sets. """
    # Randomize data indices
    num_examples = len(dataset)
    shuffled_indices = np.random.permutation(num_examples)
    
    # Calculate split index
    train_size = int(num_examples * train_ratio)
    
    # Split indices
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]
    
    # Create subset datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    return train_dataset, val_dataset

def remove_chars(row):
    text = row['text']
    
    # This pattern matches words that start with a capital letter followed by lowercase letters
    pattern = r'\b[A-Z][a-z]+\b'
    
    # Function to replace matched words
    def replace_func(match):
        word = match.group(0)
        # Return both the original and lowercase version of the word
        return word + ' ' + word.lower()

    # Apply the replacement function to words matching the pattern
    modified_text = re.sub(pattern, replace_func, text)
    
    # Now, replace punctuation marks
    modified_text = modified_text.replace(",", " <comma>").replace(".", " <dot>")
    modified_text = modified_text.replace("?", "").replace("!", "").replace("¡", "").replace("¿", "")
    
    return modified_text


# TODO: use existing implementation of word-level tokenization, e.g.
# tokenizers.models.WordLevel
# See https://github.com/huggingface/tokenizers/issues/244 for help

os.environ['USE_TORCH'] = 'TRUE'
os.environ['USE_TENSORFLOW'] = 'FALSE'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-7, help='learning rate')
parser.add_argument('--desc', type=str, help='description (without /\'s)')
parser.add_argument('--tgt_len', type=int, default=128, help='predict & backpropagate over this many tokens')
parser.add_argument('--batch_size', type=int, default=4, help='predict & backpropagate over this many tokens')

args = parser.parse_args()

#CHECKPOINT = 'openai-gpt'
CHECKPOINT = "/data/brunobian/languageModels/clm-spanish"
UNK = '<unk>'
COMMA = '<comma>'
DOT = '<dot>'

BLOCK = args.tgt_len+1 # this is actually the length of "BPTT", in tokens

# Specify the path to the "Tesis" folder in your Google Drive
local_dataset_path = '/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/data/'

# Load the dataset from local storage
#wikitext_dataset = load_from_disk(local_dataset_path)
wikitext_dataset = load_dataset('large_spanish_corpus', name='all_wikis', cache_dir= local_dataset_path,
                                split='train[:15%]', trust_remote_code=True)
#wikitext_dataset.save_to_disk(local_dataset_path)
# Build up vocabulary
vocab = set()

# Iterate through each row in the dataset
for row in wikitext_dataset:
    # TODO: genrar una funcion que usemos cada vez que carguemos un texto (ver si heredamos de algun tokenizador de HF)
    clean_text = remove_chars(row)
    words = clean_text.split(' ')
#    vocab.update(words)
    vocab.update(word for word in words if word.isalnum())

vocab = sorted(list(vocab))
# TODO: parametrizar estos tokens que los vamos a necesitar despues en la iniciacilizaciòn
vocab.append(UNK) # https://huggingface.co/transformers/model_doc/gpt.html#openaigpttokenizer
#hacer append de los signos de puntuación -- pensar lista : '.', ','
#Como estamos eliminando los tokens <comma> y <dot> cuando hacemos el isalnum() entonces los agregamos ahora
vocab.append(COMMA)
vocab.append(DOT)
word2int = {vocab[i]: i for i in range(len(vocab))}

def encode(l):
    return [(word2int[w] if w in word2int else word2int[UNK]) for w in l.split()]

def encode_batch(seqs):
    return {'input_ids': [encode(l) for l in seqs]}

# Tokenize data
dataset = wikitext_dataset.map(lambda x: encode_batch(x['text']), batched=True, num_proc=4,
                               remove_columns=['text'])

def truncate_text(examples):
    # Truncate examples so that they are evenly divisible by BPTT
    result = {
        k: [i[:(len(i) // BLOCK) * BLOCK] for i in t] \
        for k, t in examples.items()
    }
    return result

dataset = dataset.map(truncate_text, batched=True, num_proc=1)

def chunk_text(examples):
    # Split text into chunks of `BLOCK` tokens (BPTT/sequence length).
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We're truncating the entry if it's not divisible by `BLOCK`
    total_length = (total_length // BLOCK) * BLOCK
    result = {
        k: [t[i : i + BLOCK] for i in range(0, total_length, BLOCK)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

dataset = dataset.filter(lambda x: len(x['input_ids']) > 0) # remove empty sequences (incl. those less than BPTT length)
dataset = dataset.map(chunk_text, batched=True, batch_size=4) # break sequences into chunks of length `BLOCK`

# check
print(dataset)
lm_datasets = dataset

# load pretrained model
#model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
#TODO Implementar para Llama-2
model = GPT2LMHeadModel.from_pretrained('DeepESP/gpt2-spanish')
tokenizer = GPT2Tokenizer.from_pretrained('DeepESP/gpt2-spanish', use_fast= True)

# -- INICIALIZA LOS EMBEDDINGS DE LAS PALABRAS NUEVAS CON EL PROMEDIO DE LOS TOKENS SUBPALABRA ORIGINALES --
# interpolate BPE
embs = torch.zeros((len(vocab), model.config.n_embd)) # Creando nueva matriz de embeddings, por ahora en cero
for c, w in enumerate(vocab): # Recorre la lista de palabras nuevas (vocab)
    ids = tokenizer(' ' + w)['input_ids'] # Aca le agrega un espacio adelante porque el tokenizador de GPT2 funciona asi
    if len(ids) == 0:
        print('zeroed', w)
        continue
    subembs = torch.stack([model.transformer.wte.weight.data[i] for i in ids], axis = 0) # trae de el modelo los embeddings estáticos
    embs[c] = torch.mean(subembs, axis = 0) # Creo el embedding nuevo con el promedio de los embeddings subpalabra

# TODO: a mano buscar el embedding de punto y de coma original para meterlo en las pos de <dot> y <comma>
#embs[vocab.index(UNK)] = model.transformer.wte.weight.data[tokenizer.vocab[tokenizer.unk_token]]
tokenizer_vocab = tokenizer.get_vocab()
embs[vocab.index(UNK)] = model.transformer.wte.weight.data[tokenizer_vocab[tokenizer.unk_token]]
embs[vocab.index(COMMA)] = model.transformer.wte.weight.data[tokenizer_vocab[',']]
embs[vocab.index(DOT)] = model.transformer.wte.weight.data[tokenizer_vocab['.']]
#assert len(set(embs)) == len(vocab)

# update embedding layer
print('Previous vocab size:', model.transformer.wte.weight.shape[0])
model.resize_token_embeddings(len(vocab)) # achica o agranda la capa de embedggins para que tenga la dimension nueva
model.transformer.wte.weight.data.copy_(embs) # mete los embeddings calculados como promedio en la capa de embeddings (pisa todo)
print('New vocab size:', model.transformer.wte.weight.shape[0])
#print(model.transformer.wte.weight.data)
#print(model.lm_head.weight.data)

'''callback to save every 10 epochs'''
class save_callback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % 1 == 0: control.should_save = True
        else: control.should_save = False
        print(state.epoch, control.should_save)

model_save_path = Path(CHECKPOINT) / 'model'
#model_save_path = Path('/content/drive/My Drive/Tesis/model_chekpoints/' + CHECKPOINT + '_word_stimulidb+reddit' + (args.desc or ''))
#model_save_path.mkdir(parents=True, exist_ok=True)

log_save_path = Path(CHECKPOINT) / 'log'

# Save the new tokenizer
tokenizer_path = Path(CHECKPOINT) / 'tokenizer'
new_tokenizer = CustomTokenizer(word2int = word2int, int2word= vocab)
new_tokenizer.save_pretrained(tokenizer_path)

# Assuming `lm_datasets` is your dataset loaded into a Dataset object
train_dataset, val_dataset = split_dataset(lm_datasets)

lm_datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

training_args = TrainingArguments(
    str(model_save_path),
    seed = 42,
    evaluation_strategy = "epoch",

    learning_rate=args.lr,
    warmup_ratio = 0.2,
    per_device_train_batch_size = args.batch_size,
    #per_device_eval_batch_size = 4,
    logging_dir= (log_save_path),

    #num_train_epochs=50,
    save_strategy = 'epoch',
    save_steps = 2,
    num_train_epochs=10,
    #save_strategy = 'steps',
    #save_steps = 16000, # ~2hr at 2 batches/sec
    save_total_limit = 20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    callbacks=[save_callback],
    tokenizer = new_tokenizer
)

trainer.train()