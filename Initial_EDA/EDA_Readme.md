# EDA Read Me Notes

1. System: Ubuntu 22.04
2. Data Repo: https://github.com/DeepPathology/MITOS_WSI_CMC
3. Data Repo: https://github.com/DeepPathology/SlideRunner.git

## Conda Environment Dependencies

- [x] conda install matplotlib
- [x] pip install -U SlideRunner
- [x] pip install SlideRunner-dataAccess
- [x] pip install fastai
- [x] conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- [x] sudo apt-get install openslide-tools
- [x] pip install openslide-python

## SlideRunner

SliderRunner Demo walkthrough notebook shows that human readable descriptions were provided along with the annotations. However these human readable annotations do not describe in such a way as to report to a patient. 
