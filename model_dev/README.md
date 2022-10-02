## Model Iteration
### Image Size: 128x128
- [x] Resnet50- Accuracy: 72.89%
- [x] Vgg16- Accuracy: 67.98%
- [x] Alexnet- Accuracy: 50.47%
- [ ] EfficientNet-
- [ ] Resnet152-

### Image Size: 224x224

- [x] Resnet50(SGD)- Accuracy: 72.93%
- [x] Resnet50(Adam)- Accuracy: 72.99%
- [x] Resnet50(SGD, ImageNetV2 weights)- Accuracy: 73.57%


#### Learning Rate Adjustment to 0.0001:

- [x] VGG16(SGD)- Accuracy: 73.33%


#### Reduced Data:

- [x] VGG16(SGD, LR=0.0001): 71.25% 


### Image Size: 64x64

#### Reduced Dataset:
- [x] Resnet18(Pretrained Imagenet_V1, SGD, LR=0.001): 83.13%
- [x] Resnet34(SGD, LR=0.001): 81.42%
- [x] Resnet50(SGD, LR=0.001): 76.15%
- [x] Resnet50(Pretrained Imagenet_V2, SGD, LR=0.001): 84.18% 

#### Full Dataset:
- [x] Resnet50(Pretrained Imagenet_V2, SGD, LR=0.001): 87.75%
- [x] VGG16(Pretrained Imagenet_V2, SGD, LR=0.001): 88.33%
- [x] HF Vision Tranformer(ViTModel, Adam, LR=2e-5): ~90%

### Notes

- Dataset images are 128x128. Models were expecting 224x224. Changed model setup to accept 128x128 images. Possible to resize images to 224x224
- Dataset used ImageFolder class. The images used no bounding boxes or labels other than directory structure. 
- Trained for 15 Epochs which took roughly 30 minutes per model.
- Trained for 50 epochs, no big change in performance
- Train for 50 epochs and smaller learning rate, slightly improvement in performance
- Investigate 'REDUCELRONPLATEAU' implementation 
