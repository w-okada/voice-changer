FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y emacs git curl python3-pip libsndfile1-dev ffmpeg


RUN pip install torch==1.13.1
RUN pip install torchaudio==0.13.1
RUN pip install pydub==0.25.1
RUN pip install tqdm==4.64.1
RUN pip install librosa==0.9.2
#RUN pip install librosa==0.8.1
RUN pip install psutil==5.9.4
#RUN pip install psutil==5.4.8
RUN pip install tensorboard==2.11.2
RUN pip install pytz==2022.7.1
RUN pip install pyworld==0.3.2
RUN pip install retry==0.9.2
#RUN pip install numpy==1.21.6
RUN pip install h5py==3.8.0
#RUN pip install h5py==3.1.0
RUN pip install matplotlib==3.6.3
#RUN pip install matplotlib==3.2.2

RUN pip install onnx==1.13.0
RUN pip install onnxruntime==1.14.0
RUN pip install onnxsim==0.4.17

ADD /warmup.py /
RUN python3 warmup.py

ADD dummy /

RUN git clone -b v1.5.0.0_SiFiGAN https://github.com/isletennos/MMVC_Trainer.git 
WORKDIR /MMVC_Trainer/
RUN git checkout 0fa4662a8741a127cc0bdb6ccdecfc17bdcba709


RUN git clone https://github.com/isletennos/MMVC_Client.git
WORKDIR /MMVC_Trainer/MMVC_Client
RUN git checkout 461cb231b57cbb17243110eaac8435d9cca24a26

WORKDIR /
# ↓ テストスクリプトはTrainerのrootに置くとmodelsがconflictする。
ADD /scripts /MMVC_Trainer/MMVC_Client/python/
ADD /model/D_v15_best.pth /MMVC_Trainer/fine_model/
ADD /model/G_v15_best.pth /MMVC_Trainer/fine_model/

RUN cp -r /MMVC_Trainer/configs /MMVC_Trainer/configs_org

WORKDIR /MMVC_Trainer/
