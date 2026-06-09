# Tokenizing a text file
# Optional: retrieve the file ("The Verdict" by Edith Wharton)
# import urllib.request

# url = ("https://raw.githubusercontent.com/rasbt/"
#         "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#         "the-verdict.txt"
# )

import re

file_path = "the-verdict.txt"

# urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Add tokens to vocabulary with int IDs
all_words = sorted(set(preprocessed))
vocab = {token:integer for integer, token in enumerate(all_words)}


class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text


tokenizer = Tokenizer(vocab)
text = """
"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))