
import numpy as np
import pickle

# GloVe github: https://github.com/stanfordnlp/GloVe/blob/master/eval/python/evaluate.py 
# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

glove_path = "data/glove"

def load_embeddings():

    idx = 0
    word2idx = {}
    vectors = []

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()

            word = line[0]
            word2idx[word] = idx

            vector = np.array(line[1:]).astype(float).reshape((1, -1))
            vectors.append(vector)

            idx += 1
    
    f_glove = open(f'{glove_path}/glove_embeddings.npy', "wb")
    glove = np.concatenate(vectors)
    print("glove shape: ", glove.shape)
    np.save(f_glove, glove)

    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
    

def get_glove_embedding(text):
    
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
    glove = np.load(f'{glove_path}/glove_embeddings.npy')

    embeddings = []
    for word in text:
        embeddings.append(glove[word2idx[word]])
    
    attribute_embedding = np.concatenate(embeddings)
    return attribute_embedding

# get embeddings
# load_embeddings()

import json
f_subset = open("data/subset/img_ids.json", "r")
subset = json.load(f_subset)
# print(len(subset))

attribute_file = open("../data-of-multimodal-sarcasm-detection/extract/extract_all.txt", "r")
embeddings = []
counter = 0
for line in attribute_file.readlines():
    img_id = line[2:20]
    if img_id in subset:
        text = [word[1:-1] for word in (line[:-2].split(', '))[1:]]
        embeddings.append(get_glove_embedding(text).reshape((1, -1)))
    
        counter +=1
        if counter %1000 == 0:
            print(counter)

attribute_embeddings = np.concatenate(embeddings)
print(attribute_embeddings.shape)

f_attribute_embeddings = open("data/subset/attribute_embeddings.npy", "wb")
np.save(f_attribute_embeddings, attribute_embeddings)