from os import truncate
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

import json
f_subset = open("data/subset/img_ids.json", "r")
subset = json.load(f_subset)

text_file = open("../data-of-multimodal-sarcasm-detection/text/train.jsonl", "r")
counter = 0

sentences = []
for line in text_file.readlines():
    img_id = line[2:20]
    if img_id in subset:
        sentences.append(line[24:-6])
        counter +=1
        if counter %1000 == 0:
            print(counter)
        
        if counter == 1000:
            break

# https://huggingface.co/transformers/v1.0.0/quickstart.html

encoded = tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(**encoded)

    # PyTorch-Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    embeddings = outputs[0]

embeddings = torch.mean(embeddings, dim=1)
print(embeddings.shape)

f_text = open("data/subset/text_embeddings.npy", "wb")
np.save(f_text, embeddings)
f_text.close()

