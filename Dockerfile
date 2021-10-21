FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04

# Copy local file
RUN mkdir /app
WORKDIR /app
COPY . /app

# Update system
RUN apt update
RUN apt upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Install python 3.8
RUN DEBIAN_FRONTEND="noninteractive" apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.7 python3-pip

# Install python tools
RUN python3.7 -m pip install --upgrade setuptools pip distlib

# Install requirements
RUN python3.7 -m pip install -r requirements.txt
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir /app/runs

CMD python3.7 train.py --img 640 --batch 16 --epochs 300 --data custom_dataset.yaml --weights yolov5s.pt --cache
