#----------------------------------------------------------------------------
#                                  Imports
#----------------------------------------------------------------------------

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision
# from torchvision import models
# from torchvision import transforms
# from torchvision.models import resnet50
# import torchvision.models as models


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import vgg16_model_streamlit
from vgg16_model_streamlit import *

# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# from captum.attr import IntegratedGradients
# from captum.attr import GradientShap
# from captum.attr import Occlusion
# from captum.attr import NoiseTunnel
# from captum.attr import visualization as viz

#----------------------------------------------------------------------------
#                         Function Definitions
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#                         Load Model State Dictionary
#----------------------------------------------------------------------------

# def load_model(file_path):
    
#     model = models.vgg16(weights=None)

#     num_features = model.classifier[6].in_features
#     features = list(model.classifier.children())[:-1]
#     features.extend([nn.Linear(num_features, 2)])
#     model.classifier = nn.Sequential(*features)

#     model.load_state_dict(torch.load(file_path))
    
#     return model


#----------------------------------------------------------------------------
#                         Title
#----------------------------------------------------------------------------

st.title('Mitotic Figure Detection')

device = get_device()
st.write('Device used is:', device )

#----------------------------------------------------------------------------
#                         Upload Image
#----------------------------------------------------------------------------

file = st.file_uploader('Select Image File')

if file:
    image = Image.open(file)

    st.image(image)



#----------------------------------------------------------------------------
#                         Load Model State Dictionary
#----------------------------------------------------------------------------
model_pth = st.file_uploader('Select Model File')

st.title('Mitotic Figure Detection Results')

done = False

if model_pth:

    model = load_model(model_pth)
    input, pred_label_idx, predicted_label, prediction_score = load_image(file, model, device)
    st.write(predicted_label)
    st.write(prediction_score.squeeze().item())

    def batch_predict(images):

        preprocess_transform = get_preprocess_transform()

        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)
        
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    done = True

#----------------------------------------------------------------------------
#                         Model Interpretability
#----------------------------------------------------------------------------

st.title('Model Interpretability')

lime_done = False

if done == True:
    #lime_exp(model, file, device)
    img = Image.open(file)
    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                            batch_predict, # classification function
                                            top_labels=5, 
                                            hide_color=0, 
                                            num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)

    img_0 = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
    st.image(img_0)

    img_1 = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
    st.image(img_1)

    lime_done = True

#----------------------------------------------------------------------------
#                         Captum Explainability
#----------------------------------------------------------------------------

st.title('Captum Interpretability')


if lime_done == True:

    transformed_img = img_trans(file)

    with torch.no_grad():
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(input, target=1)
        

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(input,
                                        strides = (3, 8, 8),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,15, 15),
                                        baselines=0)    

    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        #["original_image", "heat_map"],
                                        ["original_image", "blended_heat_map"],
                                        ["all", "all"],
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        )
    st.pyplot()
    

    

    