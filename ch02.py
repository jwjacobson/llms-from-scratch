# Tokenizing a text file
# Optional: retrieve the file ("The Verdict" by Edith Wharton)
# import urllib.request

# url = ("https://raw.githubusercontent.com/rasbt/"
#         "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#         "the-verdict.txt"
# )

import tiktoken
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
vocab["<|unk|>"] = 1130

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)

        return text


# Verify tokenizer behavior with unknown words
# tokenizer = Tokenizer(vocab)
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(tokenizer.decode(tokenizer.encode(text)))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# strings = tokenizer.decode(integers)
# print(strings)

text2 = "Akwirw ier"
new_tokens = tokenizer.encode(text2)
for token in new_tokens:
    print(tokenizer.decode([token]))