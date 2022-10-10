# Clustering of word embeddings
# For each sample, cluster word embeddings, then
# find the emotion for each cluster of embeddings

from lib2to3.pgen2 import token
from matplotlib.pyplot import xcorr
import numpy as np
import pandas as pd
from sklearn import metrics
import scipy
from sklearn.model_selection import train_test_split
import random
from statistics import mode

import datasets
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
import torch
import torch.nn as nn
from torch.nn import functional as F

ID_COLUMN = 'TranscriptID'
INDEX_COLUMN = 'ClipIndex'
TRANSCRIPT_COLUMN = 'Transcript'
MAIN_EMOTION_COLUMN = 'MainEmotion'
EMOTION_MAP = {
    'Happiness': 0,
    'Sadness': 1,
    'Anger': 2,
    'Fear': 3,
    'Disgust': 4,
    'Surprise': 5
}
id2label = {idx:label for idx,label in enumerate(EMOTION_MAP.keys())}

data = 'MOSI/Data/Survey/Results/August/Cleaned/Videos/MOSI_allvideos_cleaned.csv'
device = torch.device('cpu')

# Tokenize the data
def preprocess(data_fpath):
    df = pd.read_csv(data_fpath)#.sample(n=50)#.head(30)

    # Reformat data into table:
    # TranscriptID | ClipIndex | Transcript | Emotion (majority vote)
    # Get majority vote for emotion
    videos = df.groupby([ID_COLUMN, INDEX_COLUMN])
    new_df = []
    for id_group in videos:
        video_df = id_group[1].reset_index()
        main_emotion_ratings = list(video_df[MAIN_EMOTION_COLUMN])
        random.shuffle(main_emotion_ratings)
        majority_emotion = mode(main_emotion_ratings)

        majority_label = EMOTION_MAP[majority_emotion]


        emotion_distribution = {
            'Happiness': 0,
            'Sadness': 0,
            'Anger': 0,
            'Fear': 0,
            'Disgust': 0,
            'Surprise': 0
        }

        for e in main_emotion_ratings:
            emotion_distribution[e] += 1

        new_emotions_list = list(emotion_distribution.values())
        new_emotions_list = new_emotions_list / np.sum(new_emotions_list)
        new_emotions_list = [float(x) for x in new_emotions_list]

        target_entropy = scipy.stats.entropy(new_emotions_list, base=6)

        new_row = (
            id_group[0][0], # transcript id
            id_group[0][1], # clip index
            video_df[TRANSCRIPT_COLUMN][0],
            majority_label,
            new_emotions_list,
            target_entropy
        )

        new_df.append(new_row)
    
    new_df = pd.DataFrame(new_df, columns=[
        'TranscriptID', 'ClipIndex', 'Transcript', 'Emotion', 'EmotionDist', 'Entropy'
    ])
    data_df = pd.DataFrame().assign(
        source=new_df['Transcript'],
        target=new_df['Emotion'],
        dist=new_df['EmotionDist'],
        entropy=new_df['Entropy']
    )

    return data_df

data_df = preprocess(data)

# model = AutoModel.from_pretrained(
#     'distilbert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')#'distilroberta-base')#, max_len=512)
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
    #'distilroberta-base'
#) 

emotion_classifier = pipeline("text-classification",model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

from sklearn import cluster
import matplotlib.pyplot as plt

pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
i = 0
avg_kl_loss = 0
out_df = {
    'Transcript': [],
    'ClusterDist': [],
    'TrueDist': [],
    'ClusterEntropy': [],
    'TrueEntropy': [],
    'KLDiv': []
}

for sentence, true_dist, true_entropy in zip(
    data_df['source'], data_df['dist'], data_df['entropy']):

    #sentence = 'I Love the mall!'
    tokenized_s = tokenizer.encode(sentence)
    #print(sentence)
    decoded_tokens = tokenizer.decode(tokenized_s, skip_special_tokens=True, clean_up_tokenization_spaces=True).split()

    embeddings = np.squeeze(pipe(sentence))

    # One cluster for each emotion (+ neutral???)
    agglo = cluster.AgglomerativeClustering(
        distance_threshold=0.5, n_clusters=None, compute_full_tree=True,
        affinity='cosine', linkage='average')
    result = agglo.fit_predict(embeddings)
    #result = agglo.fit_predict
    #print([x for x in zip(decoded_tokens, result)])

    word_clusters = dict()
    for c in result:
        if not c in word_clusters:
            word_clusters[c] = []
    # word_clusters = {
    #     0 : [],
    #     1 : [],
    #     2 : [],
    #     3 : [],
    #     4 : [],
    #     5 : []
    # }
    
    for word,cluster_assignment in zip(decoded_tokens, result):
        word_clusters[cluster_assignment].append(word)

    # print(sentence)
    # print(word_clusters)
    # exit(-1)
    sentence_emotions = []

    for k in word_clusters:
        cluster_emotions = []
        for w in word_clusters[k]:
            #print(w)
            emotion_distribution = {
                'Happiness': 0,
                'Sadness': 0,
                'Anger': 0,
                'Fear': 0,
                'Disgust': 0,
                'Surprise': 0
            } 
            emotion = emotion_classifier(w,)

            for e in emotion:
                if e['label'] == 'joy':
                    emotion_distribution['Happiness'] = e['score']

                elif e['label'] == 'sadness':
                    emotion_distribution['Sadness'] = e['score']

                elif e['label'] == 'anger':
                    emotion_distribution['Anger'] = e['score']

                elif e['label'] == 'surprise':
                    emotion_distribution['Surprise'] = e['score']

                elif e['label'] == 'fear':
                    emotion_distribution['Fear'] = e['score']

                elif e['label'] == 'disgust':
                    emotion_distribution['Disgust'] = e['score']

            #print(w, emotion_distribution)
            max_index = np.argmax(list(emotion_distribution.values()))
            majority_label = list(emotion_distribution.keys())[max_index]
            #print(emotion)
            #print(majority_label)
            cluster_emotions.append(majority_label)
        
        #print(cluster_emotions)
        if cluster_emotions:
            main_cluster_emotion = mode(cluster_emotions)
            sentence_emotions.append(main_cluster_emotion)
        #exit(-1)
   
    #print(word_clusters)
    #print(sentence_emotions)

    sentence_emotions_dist = {
        'Happiness': 0,
        'Sadness': 0,
        'Anger': 0,
        'Fear': 0,
        'Disgust': 0,
        'Surprise': 0
    }

    for e in sentence_emotions:
        sentence_emotions_dist[e] += 1

    sentence_emotions_list = list(sentence_emotions_dist.values())
    sentence_emotions_list = sentence_emotions_list / np.sum(sentence_emotions_list)
    sentence_emotions_list = [float(x) for x in sentence_emotions_list]
    
    e_dist = np.array(sentence_emotions_list)
    true_dist = np.array(true_dist)

    cluster_entropy = scipy.stats.entropy(e_dist, base=6)

    kl_loss = scipy.stats.entropy(
        pk=np.where(e_dist==0, 1e-10, e_dist),
        qk=np.where(true_dist==0, 1e-10, true_dist), base=6)
    
    avg_kl_loss += kl_loss

    out_df['Transcript'].append(sentence)
    out_df['ClusterDist'].append(e_dist)
    out_df['TrueDist'].append(true_dist)
    out_df['ClusterEntropy'].append(cluster_entropy)
    out_df['TrueEntropy'].append(true_entropy)
    out_df['KLDiv'].append(kl_loss)

    if i % 30 == 0:
        # print('Cluster entropy: ', cluster_entropy)
        # print('Actual entropy: ', true_entropy)
        # print('KL loss: ', kl_loss)

        fig = plt.figure()
        plt.plot(list(range(e_dist.shape[0])), e_dist, color='blue', label='Cluster')
        plt.plot(list(range(true_dist.shape[0])), true_dist, color='red', label='True')
        plt.title('Predicted and Original Distributions')
        plt.xticks([0,1,2,3,4,5],
            ['Happiness', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise'])
        plt.xlabel('Emotion')
        plt.ylabel('Probability')
        plt.legend(title='KL Div: ' + str(round(kl_loss,3)))
        #plt.show()

        fname = 'Models/ExperimentalApproaches/ClusterPlots/P' + str(i)
        fig.savefig(fname, bbox_inches='tight')

        #print('\n')
    #exit(-1)
    i+=1

avg_kl_loss /= len(data_df['source'])
print('Avg kl div: ', avg_kl_loss)

out_df = pd.DataFrame(out_df)
out_df.to_csv('Models/ExperimentalApproaches/ClusterResults.csv', index=False)

