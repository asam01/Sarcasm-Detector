# https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d

# Importing the relevant modules
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
# Loading the pre-trained BERT model
###################################
# Embeddings will be derived from
# the outputs of this model
model = BertModel.from_pretrained('bert-base-uncased',
           output_hidden_states = True,)
# Setting up the tokenizer
###################################
# This is the same tokenizer that
# was used in the model to generate
# embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)

# Text corpus
##############
# These sentences show the different
# forms of the word 'bank' to show the
# value of contextualized embeddings

train_file = open("../data-of-multimodal-sarcasm-detection/text/train.txt", "r")
train_dict = {}
for line in train_file.readlines():
    train_dict[line[2:20]] = int(line[-3])

attributes_file = open("../data-of-multimodal-sarcasm-detection/extract/extract_all", "r")
attribute_list = []
counter = 0
for line in attributes_file.readlines():
    img_id = line[2:20]
    if img_id in train_dict:
        attributes = [attribute[1:-1] for attribute in (line[:-2].split(', '))[1:]]
        attributes = " ".join(attributes)
        attribute_list.append((attributes, train_dict[img_id]))
        
        counter +=1
    if counter == 2000:
        break

texts = [attributes for (attributes, _) in attribute_list]


# Getting embeddings for the target
# word in all given contexts
target_word_embeddings = []

for text in texts: #ah this is wrong
    print("text: ", text)
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
    token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    
    print(token_embeddings.shape)
    break