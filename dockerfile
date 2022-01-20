FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

WORKDIR /app
RUN apt-get update
RUN apt-get install -y software-properties-common tzdata
ENV TZ=Asia/Tokyo 
RUN apt-get -y install python3.9 python3.9-distutils python3-pip
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt