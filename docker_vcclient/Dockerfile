# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# ARG DEBIAN_FRONTEND=noninteractive

# RUN apt-get update 
# RUN apt-get install -y python3-pip git

# RUN pip install torch==1.13.1
# RUN pip install numpy==1.24.1
# RUN pip install python-socketio==5.7.2

# RUN pip install fastapi==0.89.1
# RUN pip install python-multipart==0.0.5
# RUN pip install uvicorn==0.20.0
# RUN pip install websockets==10.4
# RUN pip install pyOpenSSL==23.0.0
# RUN pip install onnxruntime-gpu==1.13.1

# RUN git clone https://github.com/isletennos/MMVC_Client.git
# RUN cd MMVC_Client && git checkout 04f3fec4fd82dea6657026ec4e1cd80fb29a415c && cd -


# WORKDIR /
# ADD dummy /

# RUN git clone --depth 1 https://github.com/w-okada/voice-changer.git


#########
# Prod  #
#########
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update \
        && apt-get install -y python3-pip espeak gosu libsndfile1-dev emacs git \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN pip install uvicorn==0.21.1
RUN pip install pyOpenSSL==23.1.1
RUN pip install numpy==1.23.5
RUN pip install torch==1.13.1
RUN pip install resampy==0.4.2
RUN pip install python-socketio==5.8.0
RUN pip install fastapi==0.95.1
RUN pip install python-multipart==0.0.6
RUN pip install onnxruntime-gpu==1.14.1
RUN pip install pyworld==0.3.3
RUN pip install scipy==1.10.1
RUN pip install matplotlib==3.7.1
RUN pip install fairseq==0.12.2
RUN pip install websockets==11.0.2
RUN pip install praat-parselmouth==0.4.3
RUN pip install faiss-cpu==1.7.3
RUN pip install torchcrepe==0.0.18
RUN pip install librosa==0.10.0.post1
RUN pip install gin==0.1.6
RUN pip install gin_config==0.5.0
RUN pip install einops==0.6.0
RUN pip install local_attention==1.8.5
RUN pip install websockets==11.0.2

WORKDIR /
ADD dummy /

RUN git clone https://github.com/w-okada/voice-changer.git -b v.1.5.2


ADD /setup.sh  /voice-changer/server
ADD /exec.sh  /voice-changer/server

RUN chmod 0777 /voice-changer/server
WORKDIR /voice-changer/server

RUN git clone https://github.com/isletennos/MMVC_Client.git MMVC_Client_v13
RUN cd MMVC_Client_v13 && git checkout 04f3fec4fd82dea6657026ec4e1cd80fb29a415c && cd -

RUN git clone https://github.com/isletennos/MMVC_Client.git MMVC_Client_v15
RUN cd MMVC_Client_v15 && git checkout 461cb231b57cbb17243110eaac8435d9cca24a26 && cd -

RUN git clone https://github.com/StarStringStudio/so-vits-svc.git so-vits-svc-40
RUN git clone https://github.com/StarStringStudio/so-vits-svc.git so-vits-svc-40v2
RUN cd so-vits-svc-40v2 && git checkout 08c70ff3d2f7958820b715db2a2180f4b7f92f8d && cd -
RUN git clone https://github.com/yxlllc/DDSP-SVC.git DDSP-SVC
RUN git clone https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI.git RVC

ENTRYPOINT ["/bin/bash", "setup.sh"]
