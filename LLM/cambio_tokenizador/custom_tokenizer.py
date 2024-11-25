import os
import json

from transformers import PreTrainedTokenizer


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, word2int, int2word, unk_token="<unk>", pad_token="<pad>", **kwargs):
        self.vocab = word2int          
        self.int2word = int2word  
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Ensure pad_token is in vocab
        if pad_token not in self.vocab:
            self.vocab[pad_token] = len(self.vocab)  # Assign new ID for pad_token
            self.int2word.append(pad_token)

        # Ensure unk_token is in vocab
        if unk_token not in self.vocab:
            self.vocab[unk_token] = len(self.vocab)  # Assign new ID for unk_token
            self.int2word.append(unk_token)

        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)


    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_pretrained(cls, save_directory, filename_prefix=None):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        config_file = os.path.join(save_directory, "tokenizer_config.json")

        # Cargar el vocabulario 
        with open(vocab_file, 'r', encoding='utf-8') as f:
            word2int = json.load(f)

        # Obtener int2word 
        int2word = {v: k for k, v in word2int.items()}

        # Cargar la configuración 
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Extraer detalles de la configuración 
        unk_token = config.get('unk_token', '<unk>')
        clean_up_tokenization_spaces = config.get('clean_up_tokenization_spaces', True)

        # Inicializar tokenizador con detalles de configuración 
        return cls(word2int=word2int, int2word=int2word, unk_token=unk_token, clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)

    def tokenize(self, text, **kwargs):
        #TODO ver si se puede usar tokenización de Spacy
        text = self.pretokenize(text)
        return text.split()

    def pretokenize(self, text):
        clean_text = text.lower().replace(",", " <comma>").replace(".", " <dot>")
        clean_text = clean_text.replace("?", "").replace("!", "").replace("¡", "").replace("¿", "")

        return clean_text

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            tokens = []
            for id in ids:
                token = self.int2word[id] if id < len(self.int2word) else self.unk_token
                # Skip tokens that are special (e.g., padding or unknown, based on your tokenizer's setup)
                if token not in [self.pad_token, self.unk_token]:
                    tokens.append(token)
        else:
            tokens = [self.int2word[id] if id < len(self.int2word) else self.unk_token for id in ids]
        return tokens

    def get_vocab(self):
        return self.vocab

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]
        