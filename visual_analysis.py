import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict 

# https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-easy-way-62631c7fa8f6 

from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import model_urls

class IntResNet(ResNet): #subclass of ResNet
    def __init__(self, **kwargs):
        
        num_blocks_per_layer = [2, 2, 2, 2]
        super().__init__(BasicBlock, num_blocks_per_layer)
        
        last_layer = "avgpool" #stop at the layer before fc
        self._layers = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == last_layer: 
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

# ResNet source code: 
# https://github.com/pytorch/vision/blob/6e203b44098c3371689f56abc17b7c02bd51a261/torchvision/models/resnet.py#L252

model = IntResNet()
# load pretrained weights 
state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-5c106cde.pth") # load weights (resnet18)
model.load_state_dict(state_dict)

from PIL import Image
from torchvision import transforms

def preprocess_image(filename):
    try:
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image) # preprocess
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch
    except:
        print("bad file: ", filename)
        return None


def get_embedding(input_batch):
  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)
    
  return output


import glob
import shutil
import random

path = "data/dataset_image/dataset_image/*"
image_tensors = []
labels = []
count = 0
fnames = glob.glob(path)
random.shuffle(fnames)

train_file = open("../data-of-multimodal-sarcasm-detection/text/train.txt", "r")
train_list = {}
for line in train_file.readlines():
    train_list[line[2:20]] = int(line[-3])

for fname in fnames:
    img_id = (fname.split('/')[-1])[:-4]

    if img_id in train_list:
        image_tensor = preprocess_image(fname)
        if image_tensor != None:
            image_tensors.append(image_tensor)
            labels.append(train_list[img_id])

            # shutil.copyfile(fname, "data/subset_raw/"+img_id+".jpg")

            count +=1
            if count%1000 == 0:
                print(count)
            if count == 2000:
                break

X = torch.cat(image_tensors, dim=0)
print(X.shape)

import numpy as np
f_labels = open("data/subset/labels.npy", "wb")
labels_numpy = np.array(labels)
print(labels_numpy.shape)
np.save(f_labels, labels_numpy)


batch_size = 16
embedding_list = []
for i in range(0, len(X), batch_size):
  batch_image = X[i:i+batch_size]
  batch_embedding = get_embedding(batch_image)

  embedding_list.append(batch_embedding)

embeddings = torch.cat(embedding_list, dim=0)
print(embeddings.shape)

f_embeddings = open("data/subset/embeddings.npy", "wb")
# np.save(f_embeddings, embeddings.numpy())