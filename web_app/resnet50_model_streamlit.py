#----------------------------------------------------------------------------
#                                  Imports
#----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import cv2 as cv
import sys
import os
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

#----------------------------------------------------------------------------
#                                  CUDA or CPU
#----------------------------------------------------------------------------

def get_device():
    '''
    Returns CUDA or CPU Device
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
    print("Using " + torch.cuda.get_device_name(device))

#----------------------------------------------------------------------------
#                         Load Model State Dictionary
#----------------------------------------------------------------------------

#file_path='../model_dev/vgg16_ft.pth'

def rn50_load_model(file_path):

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = torch.load(file_path)
    # model.load_state_dict(torch.load(file_path))
    
    return model

    # model = models.resnet50(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    # model = torch.load('../model_dev/model_rn50_v2_ft.pth')
#----------------------------------------------------------------------------
#                         Define Image Transform Function
#----------------------------------------------------------------------------

def img_transform(img):
    '''
    Image transform function
    '''
#     transform = transforms.Compose([
#      transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor()
#     ])

#     transform_normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#      )
    
#     img = Image.open(path)
#     transformed_img = transform(img)
#     input = transform_normalize(transformed_img)
#     input = input.unsqueeze(0)

#     return input

# def get_input_tensors(img):
#     transf = get_input_transform()
#     # unsqeeze converts single image to batch of 1
#     return transf(img).unsqueeze(0)

#def get_input_transform(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    #transf_img = transf(img).unsqueeze(0)
    return transf
    
#     return transf

def get_input_tensors(img):
     transf = img_transform(img)
#     # unsqeeze converts single image to batch of 1
     return transf(img).unsqueeze(0)
#----------------------------------------------------------------------------
#                         Define Image Transform Function
#----------------------------------------------------------------------------

def rn50_load_image(img, model, device):
    '''
    Load image into model
    '''
    img_input = get_input_tensors(img)
    model.eval()
    model = model.to(device)
    img_cuda_input = img_input.to(device)

    
    output = model(img_cuda_input) #.to(device)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = str(pred_label_idx.item())
    
    return img_cuda_input, output, prediction_score, predicted_label, pred_label_idx
    #print(output)
    #print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    
#----------------------------------------------------------------------------
#                         Define Image Transform Function
#----------------------------------------------------------------------------    
    
#img_path = '../model_dev/Data_CMC_COADEL_224_1/val/Mitosis/10003.jpg'

def get_image(path):
    '''
    Open up the image and display it
    '''
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

#----------------------------------------------------------------------------
#                         PIL transform function
#----------------------------------------------------------------------------         
        
        
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

#----------------------------------------------------------------------------
#                         Image Preprocessing
#----------------------------------------------------------------------------    


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf 

#----------------------------------------------------------------------------
#                         Batch Prediction
#---------------------------------------------------------------------------- 

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

#----------------------------------------------------------------------------
#                         Main execution
#----------------------------------------------------------------------------       


# path = '../model_dev/Data_CMC_COADEL_64_1/val/Mitosis/'
# model_path = '../model_dev/model_rn50_v2_ft.pth'
# file = sys.argv[1]
# img_0 = Image.open(path + file)
# img_0.show()

# img = get_image(path+file)

# device = get_device()
# model = load_model(model_path)

# load_image(img, model, device)

# pill_transf = get_pil_transform()
# preprocess_transform = get_preprocess_transform()

#----------------------------------------------------------------------------
#                         LIME Explainability
#----------------------------------------------------------------------------  

# explainer = lime_image.LimeImageExplainer()
# explanation = explainer.explain_instance(np.array(pill_transf(img_0)), 
#                                          batch_predict, # classification function
#                                          top_labels=5, 
#                                          hide_color=0, 
#                                          num_samples=1000)

# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
# img_boundry1 = mark_boundaries(temp/255.0, mask)
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
# img_boundry2 = mark_boundaries(temp/255.0, mask)
         

# img = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
# img.save('res_lime_0.png')

# img = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
# img.save('res_lime_1.png')

#----------------------------------------------------------------------------
#                         Display Images
#----------------------------------------------------------------------------  

# file_1 = 'res_lime_0.png'
# file_2 = 'res_lime_1.png'
# path_1 = '../model_dev/'

# img_1 = Image.open(path_1 + file_1)
# img_2 = Image.open(path_1 + file_2)

# img_1.show()
# img_2.show()

#----------------------------------------------------------------------------
#                         End Of File
#---------------------------------------------------------------------------- 
