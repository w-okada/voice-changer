FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y python3-pip git

RUN pip install torch torchvision torchaudio
RUN pip install Cython==0.29.32
RUN pip install numpy==1.22.4 
RUN pip install scipy==1.9.0
RUN pip install librosa==0.9.2
RUN pip install phonemizer==3.2.1
RUN pip install Unidecode==1.3.4
RUN pip install resampy==0.4.0

RUN pip install tqdm==4.64.0
RUN pip install retry==0.9.2
RUN pip install psutil==5.9.1
RUN pip install python-socketio==5.7.1
RUN pip install matplotlib==3.5.3

RUN pip install fastapi==0.85.0
RUN pip install python-multipart==0.0.5
RUN pip install uvicorn==0.18.3
RUN pip install websockets==10.4
RUN pip install pyOpenSSL==23.0.0
RUN pip install onnxruntime-gpu==1.13.1

RUN git clone --depth 1 https://github.com/isletennos/MMVC_Trainer.git -b v1.3.2.2
WORKDIR /MMVC_Trainer/monotonic_align
RUN python3 setup.py build_ext --inplace
 
WORKDIR /
ADD dummy /

RUN git clone  --depth 1 https://github.com/w-okada/voice-changer.git -b v.1.3.0


#########
# Prod  #
#########
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update \
        && apt-get install -y python3-pip espeak gosu libsndfile1-dev emacs \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

### Copy from base
COPY --from=base --chmod=777 /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=base --chmod=777 /voice-changer /voice-changer
COPY --from=base --chmod=777 /MMVC_Trainer /voice-changer/server/MMVC_Trainer
RUN chmod 0777 /voice-changer/server/MMVC_Trainer
RUN chmod 0777 /voice-changer/server

ADD /setup.sh  /voice-changer/server
ADD /exec.sh  /voice-changer/server


WORKDIR /voice-changer/server
ENTRYPOINT ["/bin/bash", "setup.sh"]
CMD [ "-h"]
