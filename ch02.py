# Tokenizing a text file
# Optional: retrieve the file ("The Verdict" by Edith Wharton)
# import urllib.request
# 
# url = ("https://raw.githubusercontent.com/rasbt/"
#         "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#         "the-verdict.txt"
# )

import re

file_path = "the-verdict.txt"

# urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Remove punctuation and whitespace
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

