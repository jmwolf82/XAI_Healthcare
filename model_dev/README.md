## Model Iteration
Image Size: 128x128
- [x] Resnet50- Accuracy: 72.89%
- [x] Vgg16- Accuracy: 67.98%
- [x] Alexnet- Accuracy: 50.47%
- [ ] EfficientNet-
- [ ] Resnet152-
<br>
Image Size: 224x224
- [x] Resnet50(SGD)- Accuracy: 72.93%
- [x] Resnet50(Adam)- Accuracy: 72.99%
- [x] Resnet50(SGD, ImageNetV2 weights)- Accuracy: 

### Notes

- Dataset images are 128x128. Models were expecting 224x224. Changed model setup to accept 128x128 images. Possible to resize images to 224x224
- Dataset used ImageFolder class. The images used no bounding boxes or labels other than directory structure. 
- Trained for 15 Epochs which took roughly 30 minutes per model. 
