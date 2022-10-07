FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
ADD https://xai-healthcare.s3.amazonaws.com/vgg16_ft.pth .
COPY . /usr/app
EXPOSE 80
WORKDIR /usr/app
USER root
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
CMD streamlit run vgg16_streamlit.py --server.port 80 --server.maxUploadSize 1024 --deprecation.showPyplotGlobalUse false