# Initial EDA Notes

1. System: Ubuntu 22.04
2. Data Repo: https://github.com/DeepPathology/MITOS_WSI_CMC
3. Tool Repo: https://github.com/DeepPathology/SlideRunner.git

## Conda Environment Dependencies

- [x] create seperate conda environment
- [x] conda install matplotlib
- [x] pip install -U SlideRunner
- [x] pip install SlideRunner-dataAccess
- [x] pip install fastai
- [x] conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- [x] sudo apt-get install openslide-tools
- [x] pip install openslide-python
- [x] pip install object-detection-fastai

## DATA: MITOS_WSI_CMC

- [x] Run Setup.ipynb to download data
- [x] Run Evaluation.ipynb
- [x] Run RetinaNet-CMC-MEL.ipynb
- [x] Run RetinaNet-CMC-ODAEL.ipynb
- [ ] Run RetinaNet-CMC-CODAEL.ipynb

## Annotations

- [x] Annotations include coordinate date 
- [x] Annotations include text annotation (description)

## TOOL: SlideRunner

- [x] Run SlideRunner Demo Walkthrough notebook
- [ ] Run SlideRunner Tool on non-demo WSI .svs file

SliderRunner Demo walkthrough notebook shows that human readable descriptions were provided along with the annotations. However these human readable annotations do not describe in such a way as to report to a patient. 

## TOOL: QuPath 

- [x] Install QuPath 
- [x] Manually look at WSI .svs files

The follow is an screen shot from the QuPath tool with a WSI displayed. 

![QuPath WSI](https://github.com/jmwolf82/XAI_Healthcare/blob/main/Initial_EDA/images/qupath_wsi_ex.tif)


## Issues/Challenges/Questions

- [x] Address Errors encountered when attempting to run notebooks
- [x] fast.ai.callbacks producing ModuleNotFoundError - fast.ai.callback fixed issue
- [x] Address NameError: name 'Patch' is not defined in RetinaNet nb's
- [x] Understand annotation/image correlation - In process refer to Pands_EDA folder






