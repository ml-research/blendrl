# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.07-py3
FROM nvcr.io/nvidia/pytorch:21.10-py3
# FROM nvcr.io/nvidia/pytorch:23.07-py3

#FROM anibali/pytorch:1.8.0-cuda10.2-ubuntu20.04

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

# Select the working directory
WORKDIR  /Workspace

# Install system libraries required by OpenCV.
#RUN apt-get update \
#    && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get update

RUN pip install --upgrade pip


# RUN apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

# RUN pip install stable-baselines3[extra]==1.7.0

# RUN opencv-python==4.8.0.74
# RUN pip install opencv-python-headless==4.5.5.62
RUN pip install gym

RUN pip install protobuf==3.20.*

RUN pip install ale-py==0.7.4

RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
