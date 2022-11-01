# import torch
# import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast


from tensorflow.keras.applications.resnet50 import ResNet50

# https://pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None)
print("loaded model")

# from PIL import Image
import tensorflow as tf

class BadImageError(Exception):
    pass

def preprocess_image(filename):
    width = 448
    height = 448
    box_width = 32
    box_height = 32
    region_tensors = []
    try:
        img = tf.io.read_file(filename)
        tensor = tf.io.decode_image(img, channels=3)
    except:
        print("bad file: ", filename)
        raise BadImageError

    tensor = tf.image.resize(tensor, [448, 448])

    for top in range(0, height, box_height):
        for left in range(0, width, box_width):
            region = tf.image.crop_to_bounding_box(tensor, top, left, box_height, box_width)

            # print(tf.equal(region, tensor[top:top+box_height,left:left+box_width,:]))

            region_tensors.append(tf.expand_dims(region, 0))

    input_batch =  tf.concat(region_tensors, 0)
    return input_batch


def get_embedding(input_batch):
  # move the input and model to GPU for speed if available
  with tf.device("/GPU:0"):
    input_batch = tf.stop_gradient(input_batch)
    output = model(input_batch)
    
  return output


import glob
import numpy as np

path = "dataset_image/*"
count = 0
fnames = glob.glob(path)
print("loaded all filenames")

embedding_list = []

for fname in fnames:
    tweet_id = (fname.split('/')[-1])[:-4]
    # print("tweet id: ", tweet_id)

    try:
        X = preprocess_image(fname)
    except BadImageError:
        continue

    # print("img shape: ", tf.shape(X))

    batch_embedding = get_embedding(X).numpy().reshape((196, 2048))
    # print("embedding shape: ", batch_embedding.shape)

    f = open("imageVector2/"+tweet_id+".npy", "wb")
    np.save(f, batch_embedding)
    f.close()
    
    count +=1
    # break # comment out later
    if count%1000 == 0:
        print(count)

# embeddings = torch.cat(embedding_list, dim=0)
# print(embeddings.shape)

# f_embeddings = open("data/subset/embeddings_postpool.npy", "wb")
# np.save(f_embeddings, embeddings.numpy())
# f_embeddings.close()
