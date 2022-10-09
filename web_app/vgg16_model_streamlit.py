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

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

#----------------------------------------------------------------------------
#                                  CUDA or CPU
#----------------------------------------------------------------------------

def get_device():
    '''
    Returns CUDA or CPU Device
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using " + torch.cuda.get_device_name(device))
    
    return device

#----------------------------------------------------------------------------
#                         Load Model State Dictionary
#----------------------------------------------------------------------------

#file_path='../model_dev/vgg16_ft.pth'

def load_model(file_path):
    
    model = models.vgg16(weights=None)

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 2)])
    model.classifier = nn.Sequential(*features)

    model.load_state_dict(torch.load(file_path))
    
    return model

#----------------------------------------------------------------------------
#                         Define Image Transform Function
#----------------------------------------------------------------------------

def img_transform(img):
    '''
    Image transform function
    '''
    transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
     )
    #img = Image.open(path)
    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    return input

#----------------------------------------------------------------------------
#                         Define Image Transform Function
#----------------------------------------------------------------------------

def img_trans(img):
    '''
    Image transform without returning an input
    '''
    transform = transforms.Compose([
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ])
    #img = Image.open(path)
    img_r = transform(img)
    return img_r

#----------------------------------------------------------------------------
#                         Define Image Transform Function
#----------------------------------------------------------------------------

def load_image(img, model, device):
    '''
    Load image into model
    '''
    img_input = img_transform(img)
    model.eval()
    model = model.to(device)
    img_input = img_input.to(device)
    img_cuda_input = img_input.to(device)

    output = model(img_input) #.to(device)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = str(pred_label_idx.item())
    print(output)
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    return img_input, pred_label_idx, predicted_label, prediction_score

    
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

# def batch_predict(images):

#     preprocess_transform = get_preprocess_transform()

#     model.eval()
#     batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     batch = batch.to(device)
    
#     logits = model(batch)
#     probs = F.softmax(logits, dim=1)
#     return probs.detach().cpu().numpy()

#----------------------------------------------------------------------------
#                         Main execution
#----------------------------------------------------------------------------       


# path = '../model_dev/Data_CMC_COADEL_224_1/val/Mitosis/'
# model_path = '../model_dev/vgg16_ft.pth'
# file = sys.argv[1]
# img_0 = Image.open(path + file)
# img_0.show()

# device = get_device()
# model = load_model(model_path)

# input, pred_label_idx = load_image(path + file, model, device)

# pill_transf = get_pil_transform()
# preprocess_transform = get_preprocess_transform()

#----------------------------------------------------------------------------
#                         LIME Explainability
#----------------------------------------------------------------------------  
def lime_exp(model, image, device):

    pill_transf = get_pil_transform()

    img = Image.open(image)
    # img_input = img_transform(image)
    img_0 = np.array(pill_transf(img))
    # model.eval()
    # model = model.to(device)
    # img_input = img_input.to(device)
    # logits = model(img_input)
    # probs = F.softmax(logits, dim=1)
    # prob = probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                            batch_predict(model, img_0), # classification function
                                            top_labels=5, 
                                            hide_color=0, 
                                            num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)

# fig = plt.figure()

# fig.add_subplot(1, 2, 1)

# plt.imshow(img_boundry1)
# plt.axis('off')
# plt.title("Positive Only")

# fig.add_subplot(1, 2, 2)

# plt.imshow(img_boundry2)
# plt.axis('off')
# plt.title("Everything")

#plt.imshow(img_boundry1)
#plt.imshow()

# img = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
# img.save('lime_0.png')

# img = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
# img.save('lime_1.png')

#----------------------------------------------------------------------------
#                         Display Images
#----------------------------------------------------------------------------  

# file_1 = 'lime_0.png'
# file_2 = 'lime_1.png'
# path_1 = '../model_dev/'

# img_1 = Image.open(path_1 + file_1)
# img_2 = Image.open(path_1 + file_2)

# img_1.show()
# img_2.show()

#----------------------------------------------------------------------------
#                         Captum Explainability
#---------------------------------------------------------------------------- 

# transformed_img = img_trans(path+file)

# with torch.no_grad():
#     integrated_gradients = IntegratedGradients(model)
#     attributions_ig = integrated_gradients.attribute(input, target=1)
    

# occlusion = Occlusion(model)

# attributions_occ = occlusion.attribute(input,
#                                        strides = (3, 8, 8),
#                                        target=pred_label_idx,
#                                        sliding_window_shapes=(3,15, 15),
#                                        baselines=0)    

# _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                       np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                       ["original_image", "heat_map"],
#                                       ["all", "positive"],
#                                       show_colorbar=True,
#                                       outlier_perc=2,
#                                      )
# attributions_occ = np.array(attributions_occ.cpu().detach().numpy())

# #img = Image.fromarray(attributions_occ)
# img = Image.fromarray(attributions_occ* 255).dtype(np.uint8)
# img.save('captum_0.png')

#----------------------------------------------------------------------------
#                         End Of File
#---------------------------------------------------------------------------- 
