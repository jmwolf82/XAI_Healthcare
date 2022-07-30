# EDA Read Me Notes

1. System: Ubuntu 22.04
2. Data Repo: https://github.com/DeepPathology/MITOS_WSI_CMC
3. Conda Environment Dependencies
- conda install matplotlib
- pip install -U SlideRunner
- pip install SlideRunner-dataAccess
- pip install fastai
- conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- sudo apt-get install openslide-tools
- pip install openslide-python
SliderRunner Demo walkthrough notebook shows that human readable descriptions were provided along with the annotations. However these human readable annotations do not describe in such a way as to report to a patient. 
