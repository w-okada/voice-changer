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

ADD /warmup.py /
RUN python3 warmup.py

ADD dummy /

RUN git clone -b v1.5.0.0_SiFiGAN https://github.com/isletennos/MMVC_Trainer.git 
WORKDIR /MMVC_Trainer/
RUN git checkout c242d3d1cf7f768af70d9735082ca2bdd90c45f3

RUN git clone https://github.com/isletennos/MMVC_Client.git
WORKDIR /MMVC_Trainer/MMVC_Client
RUN git checkout 3374a1177b73e3f6d600e5dbe93af033c36ee120


WORKDIR /
# ↓ テストスクリプトはTrainerのrootに置くとmodelsがconflictする。
ADD /scripts /MMVC_Trainer/MMVC_Client/python/
ADD /model/D_v15_best.pth /MMVC_Trainer/fine_model/
ADD /model/G_v15_best.pth /MMVC_Trainer/fine_model/

RUN cp -r /MMVC_Trainer/configs /MMVC_Trainer/configs_org

WORKDIR /MMVC_Trainer/



