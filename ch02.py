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


# Using tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# strings = tokenizer.decode(integers)
# print(strings)

# text2 = "Akwirw ier"
# new_tokens = tokenizer.encode(text2)
# for token in new_tokens:
#     print(tokenizer.decode([token]))

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# enc_text = tokenizer.encode(raw_text)
# enc_sample = enc_text[50:]

# context_size = 4

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

# torch Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt,
                        batch_size=4,
                        max_length=256,
                        stride=128,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=8, stride=2, shuffle=False
# )
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)
# second_batch = next(data_iter)
# print(second_batch)

# embedding layer
# input_ids = torch.tensor([2, 3, 5, 1])
# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# # print(embedding_layer.weight)
# # print(embedding_layer(torch.tensor([3])))
# print(embedding_layer(input_ids))

# Absolute positining
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings

print(input_embeddings.shape)