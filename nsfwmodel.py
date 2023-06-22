#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torchvision.transforms import Normalize
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import timm

#config
config = {}
config['model_name'] = 'convnext_base_in22ft1k'
config['threshold_score'] = 0.93
config['size'] = 224
config['enable_gpu'] = False
normalize_t = Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

#nsfw classifier
class NSFWClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        nsfw_model=self
        nsfw_model.root_model = timm.create_model(config['model_name'], pretrained=True)
        nsfw_model.linear_probe = nn.Linear(1024, 1, bias=False)
    
    def forward(self, x):
        nsfw_model = self
        x = normalize_t(x)
        x = nsfw_model.root_model.stem(x)
        x = nsfw_model.root_model.stages(x)
        x = nsfw_model.root_model.head.global_pool(x)
        x = nsfw_model.root_model.head.norm(x)
        x = nsfw_model.root_model.head.flatten(x)
        x = nsfw_model.linear_probe(x)
        return x

    def is_nsfw(self, img, threshold = config['threshold_score']):
        img = img.resize((config['size'], config['size']))
        img = np.array(img)/255
        img = T.ToTensor()(img).unsqueeze(0).float()
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        with torch.no_grad():
            score = self.forward(img).sigmoid()[0].item()
        return score > threshold, score


#load base model
nsfw_model = NSFWClassifier()
nsfw_model = nsfw_model.eval()


#load linear weights
linear_pth = 'nsfwmodel_281.pth'
linear_state_dict = torch.load(linear_pth, map_location='cpu')
nsfw_model.linear_probe.load_state_dict(linear_state_dict)
if config['enable_gpu']:
    nsfw_model = nsfw_model.cuda() 



#debug
if __name__ == "__main__":
    #load data
    img_pth = 'cat.jpg'
    img = Image.open(img_pth).convert('RGB')
    
    #classify
    out, score = nsfw_model.is_nsfw(img)
    print(f'is_nsfw: {out} (score: {score:.2f})')


















