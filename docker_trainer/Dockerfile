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

RUN git clone --depth 1 -b v1.5.0.0_SiFiGAN https://github.com/isletennos/MMVC_Trainer.git 
WORKDIR /MMVC_Trainer/

#RUN git checkout 1109f1201e1ee9008ee61dc221d620eb1c93d7d0
RUN git checkout f28a1514b6731c36bd68989c600bff0a3b217d75


ADD /model/D_v15_best.pth /MMVC_Trainer/fine_model/
ADD /model/G_v15_best.pth /MMVC_Trainer/fine_model/

RUN cp -r /MMVC_Trainer/configs /MMVC_Trainer/configs_org




