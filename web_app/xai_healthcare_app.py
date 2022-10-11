#----------------------------------------------------------------------------
#                                  Imports
#----------------------------------------------------------------------------

from cProfile import label
import streamlit as st
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time

import vgg16_model_streamlit
from vgg16_model_streamlit import *
import resnet50_model_streamlit
from resnet50_model_streamlit import *
import vit_model_streamlit
from vit_model_streamlit import *

import matplotlib.pyplot as plt
from pathlib import Path

#----------------------------------------------------------------------------
#                         Path to Files
#----------------------------------------------------------------------------

vgg16_path = '/home/jmwolf/repos/XAI_Healthcare/model_dev/vgg16_ft.pth'
vgg16_model = load_model(vgg16_path)
# File load debug
# st.write(vgg16_model)

rn50_path = '/home/jmwolf/repos/XAI_Healthcare/model_dev/model_rn50_v2_ft.pth'
rn50_model = rn50_load_model(rn50_path)
# File load debug
# st.write(rn50_model)

vit_path = '/home/jmwolf/repos/XAI_Healthcare/model_dev/trns_model.pt'
vit_model = vit_load_model(vit_path)
# File load debug
# st.write(vit_model)

#----------------------------------------------------------------------------
#                         Title
#----------------------------------------------------------------------------


st.title('Mitotic Figure Detection')

#----------------------------------------------------------------------------
#                         Check CUDA Device GPU vs CPU
#----------------------------------------------------------------------------
st.header("Device Verification")
device = get_device()
st.write('Device used is:', device )

#----------------------------------------------------------------------------
#                         Image Resize Function
#----------------------------------------------------------------------------

def image_resize(image):

    transform = transforms.CenterCrop((64,64))
    image_crop = transform(image)
    
    return image_crop

#----------------------------------------------------------------------------
#                         Radio Button
#----------------------------------------------------------------------------

st.header('Select a model to use')

model_sel = st.selectbox(
    'Select a model to use:',
    ('None','Resnet50','VGG16','ViT')
)

# model_sel = st.radio(
#     "Select Model to Use",
#     ('Default','Resnet50', 'VGG16', 'ViT')
#     )

if model_sel == 'Resnet50':
    model = rn50_model
    st.write('Model being used is Resnet50')

elif model_sel == 'VGG16':
    model = vgg16_model
    st.write('Model being used is VGG16')

elif model_sel == 'ViT':
    model = vit_model
    st.write('Model being used is Vision Transformer')

else:
    model = None
    st.write('No model selected')

# Radio button model selection verification debug
# st.write(model)

#----------------------------------------------------------------------------
#                         Verify Model Selected
#----------------------------------------------------------------------------

st.header('Verify Model in use')

model_print = st.button('Print Model')

if model_print:

    st.write(model)

#----------------------------------------------------------------------------
#                         Image Uploader
#----------------------------------------------------------------------------


st.header('Upload an image')

file = st.file_uploader('Select Image File')
img_upload = False

if file:
    image = Image.open(file)

    st.image(image)
    width, height = image.size
    st.write(width, height)
    img_upload = True

#----------------------------------------------------------------------------
#                 Model Predictions
#----------------------------------------------------------------------------

st.header('Model Predictions')

# model_pred = st.button('Run Prediction')

#----------------------------------------------------------------------------
#                 Resnet50 Image Resize and Prediction
#----------------------------------------------------------------------------

rn_pred = False

if img_upload == True and model_sel == 'Resnet50':
        
        rn50_img = image_resize(image)
        width, height = rn50_img.size
        # st.write(width, height)
        # st.image(rn50_img)
        input, output, prediction_score, predicted_label, pred_label_idx = rn50_load_image(rn50_img, model, device)
        
        st.write(predicted_label)
        st.write(prediction_score.squeeze().item())

        rn_pred = True

#----------------------------------------------------------------------------
#                         VGG16 Prediction
#----------------------------------------------------------------------------        

vgg_pred = False

if img_upload == True and model_sel == 'VGG16':

    model = vgg16_model
    input, pred_label_idx, predicted_label, prediction_score = load_image(image, model, device)
    st.write(predicted_label)
    st.write(prediction_score.squeeze().item())

    vgg_pred = True

#----------------------------------------------------------------------------
#                         Vision Transformer Prediction
#----------------------------------------------------------------------------  

if img_upload == True and model_sel == 'ViT':

    feature_extractor = ViTFeatureExtractor(vit_model)
    encodings = feature_extractor(images=image, return_tensors="pt")
        

    with torch.no_grad():
        feature_extractor = ViTFeatureExtractor(vit_path)
        model = vit_model   

        #inputs = feature_extractor(image, return_tensors="pt")
        # model.eval()
        inputs = torch.tensor(np.stack(feature_extractor(image)['pixel_values'], axis=0))
        
        inputs = inputs.to(device)
        output = model(inputs)
        loss, logits = model(inputs)
        out = nn.Softmax()
        pred_probab = nn.Softmax(dim=1)(loss)
        y_pred = pred_probab.argmax(1)
        
    # image_data= logits.data.cpu().numpy()
    # image_data = Image.fromarray((image_data * 255).astype(np.uint8))

    y_pred = y_pred.cpu()
    y_pred = y_pred.int()
    st.write(pred_probab)
    st.write(y_pred.item())
    if y_pred.item() == 0:
        st.write('something')  

#----------------------------------------------------------------------------
#                         LIME Interpretability
#----------------------------------------------------------------------------  

st.header('LIME Interpretability')  

lime_button = st.button('Run Lime')

#----------------------------------------------------------------------------
#                 Resnet50 LIME
#----------------------------------------------------------------------------
if rn_pred == True and lime_button:

        with st.spinner('Wait for it . . .'):
            def batch_predict(images):
                model.eval()
                batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                batch = batch.to(device)
                
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                return probs.detach().cpu().numpy()

            pill_transf = get_pil_transform()
            preprocess_transform = get_preprocess_transform()

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(np.array(pill_transf(rn50_img)), 
                                                    batch_predict, # classification function
                                                    top_labels=5, 
                                                    hide_color=0, 
                                                    num_samples=1000)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
            img_boundry1 = mark_boundaries(temp/255.0, mask)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
            img_boundry2 = mark_boundaries(temp/255.0, mask)

            # img_0 = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
            # st.image(img_0)

            img_1 = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
            st.image(img_1)
            img_1 = img_1.save("lime_vgg.png")

            with open("lime_vgg.png", "rb") as file:
                vgg_lime_dwnld = st.download_button('Download LIME Images',
                            file,
                            "lime_vgg.png",
                            mime="image/png")
        st.success('Done!')
        # st.balloons()
#----------------------------------------------------------------------------
#                         VGG16 Lime
#---------------------------------------------------------------------------- 

if vgg_pred == True and lime_button:
    
    with st.spinner('Wait for it . . .'):
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

        pill_transf = get_pil_transform()
        preprocess_transform = get_preprocess_transform()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(pill_transf(image)), 
                                                batch_predict, # classification function
                                                top_labels=5, 
                                                hide_color=0, 
                                                num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
        img_boundry1 = mark_boundaries(temp/255.0, mask)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
        img_boundry2 = mark_boundaries(temp/255.0, mask)

        # img_0 = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
        # st.image(img_0)
        # img_0 = img_0.save("vgg_lime_0.png")

        img_1 = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
        st.image(img_1)
        img_1 = img_1.save("vgg_lime_1.png")
        
        with open("vgg_lime_1.png", "rb") as file:
            vgg_lime_dwnld = st.download_button('Download LIME Images',
                            file,
                            "vgg_lime_1.png",
                            mime="image/png")
    st.success('Done!')

#----------------------------------------------------------------------------
#                         Captum Explainability
#---------------------------------------------------------------------------- 

st.header('CAPTUM Explainability')  

capt_button = st.button('Run Capt')

#----------------------------------------------------------------------------
#                         RN50 Captum
#---------------------------------------------------------------------------- 

if rn_pred == True and capt_button:
    
        with st.spinner('Wait for it . . .'):
            transformed_img = img_trans(rn50_img)

            with torch.no_grad():
                integrated_gradients = IntegratedGradients(model.cpu())
                attributions_ig = integrated_gradients.attribute(input.cpu(), target=1)
                    

                occlusion = Occlusion(model.cpu())

                attributions_occ = occlusion.attribute(input.cpu(),
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
            plt.savefig("captum_rn.png")
            st.pyplot()

            with open("captum_rn.png", "rb") as file:
                captum_dwnld = st.download_button('Download CAPT Images',
                                file,
                                "captum_rn.png",
                                mime="image/png")
        st.success('Done!')

#----------------------------------------------------------------------------
#                         VGG16 Captum
#---------------------------------------------------------------------------- 

if vgg_pred == True and capt_button:
    
    with st.spinner('Wait for it . . .'):
        transformed_img = img_trans(image)

        with torch.no_grad():
            integrated_gradients = IntegratedGradients(model.cpu())
            attributions_ig = integrated_gradients.attribute(input.cpu(), target=1)
                

            occlusion = Occlusion(model.cpu())

            attributions_occ = occlusion.attribute(input.cpu(),
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
        plt.savefig("captum_vgg.png")
        st.pyplot()

        with open("captum_vgg.png", "rb") as file:
            captum_dwnld = st.download_button('Download CAPT Images',
                            file,
                            "captum_vgg.png",
                            mime="image/png")
    st.success('Done!')

#----------------------------------------------------------------------------
#                         Image to Text
#----------------------------------------------------------------------------      

st.header('Text Based Explanation')  

text_button = st.button('Run Text Output')

if text_button:

    st.write("Feature is coming soon!")
    st.balloons()

#----------------------------------------------------------------------------
#                         End of File
#----------------------------------------------------------------------------     