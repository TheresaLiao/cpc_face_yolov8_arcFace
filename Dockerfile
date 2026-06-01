FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# regular update
RUN apt-get update
# --fix-missing && apt-get -y upgrade && apt-get autoremove

RUN apt-get install -y --no-install-recommends --fix-missing \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-pip \
    git \
    vim \
    pkg-config \
    python3-opencv 
    
RUN apt-get -y update
RUN apt-get -y install curl

COPY . /home/faceRecog_yolov8face_arcface/
WORKDIR /home/faceRecog_yolov8face_arcface/


RUN pip3 install --upgrade pip
RUN pip3 install setuptools
    

RUN pip3 install \
    pandas==1.3.5 \
    gevent==22.10.2 \
    progressbar==2.5 \
    opencv-python==4.5.5.62 \
    requests==2.28.2 \
    scikit-learn==1.2.1 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    filterpy==1.4.5 \
    torchsummary==1.5.1 \
    future==0.18.3 \
    editdistance==0.6.2 \
    protobuf==3.19.1 \
    onnx==1.10.2 \
    shapely==1.3.0 \
    tqdm==4.64.1 \
    numpy==1.23.0 \
    PyYAML==6.0\
    seaborn==0.12.2\
    Pillow \
    scikit-image \
    pymssql==2.2.7 \
    flask

RUN pip3 install \ 
    packages/typing_extensions-4.5.0-py3-none-any.whl \
    packages/torch-1.10.1+cu113-cp38-cp38-linux_x86_64.whl \
    packages/torchaudio-0.10.1+cu113-cp38-cp38-linux_x86_64.whl \
    packages/torchvision-0.11.2+cu113-cp38-cp38-linux_x86_64.whl
    

RUN cd ultralytics && pip install -e . && cd ..
