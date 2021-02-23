from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
import io
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# model_savename = 'squeeze_15ep_fe'
# path = 'checkpoints/'
# model_ft = torch.load(path + model_savename)
# model_ft.eval()

class SqueeznetClassifier():
    def __init__(
        self,
        model_path = 'checkpoints/squeeze_15ep_fe',
        data_transformer = data_transforms['val'],
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    ):
        self.model_path = model_path
        self.data_transformer = data_transformer
        self.class_names = class_names
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        #self.model.to(self.device)
        self.model.eval()

    def single_img_pred(self, img_path):
        im = Image.open(img_path)
        im_tensor = data_transforms['val'](im)#.to(self.device)
        im_tensor.unsqueeze_(0)
        
        output = self.model(im_tensor)
        _, pred = torch.max(output, 1)

        # print('predicted: {}'.format(class_names[pred]))
        return self.class_names[pred]

if __name__ == '__main__' :
    Classifier = SqueeznetClassifier()
    print(Classifier.single_img_pred('akiec.jpg'))