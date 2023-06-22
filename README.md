# Binary Classifier for NSFW Content
Binary classifier for NSFW content, coded in Pytorch and able to use GPU-acceleration.  
This model was trained for my use case because other available models had a lower accuracy on my dataset.  
The model is able to detect several types of NSFW/Explicit content. 

## Usage
You must read nsfwmodel.py to be able to test the model and to use it for your use case.  
The base model is a pretrained convnext_base_in22ft1k, the nsfw binary classifier is a linear classifier trained on features produced by the base model.
- (image) > (convnext_base_in22ft1k) > (nsfwmodel_281) > (nsfw score) > (is_nsfw?)

Demo:
```
python3 nsfwmodel.py 
```
## Requirements
- [timm](https://pypi.org/project/timm/)


  
___  
⭐ If this repo is useful to you! Thanks
