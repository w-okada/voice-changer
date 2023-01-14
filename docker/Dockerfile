FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y python3-pip git

RUN pip install torch==1.13.1
RUN pip install numpy==1.24.1
RUN pip install python-socketio==5.7.2

RUN pip install fastapi==0.89.1
RUN pip install python-multipart==0.0.5
RUN pip install uvicorn==0.20.0
RUN pip install websockets==10.4
RUN pip install pyOpenSSL==23.0.0
RUN pip install onnxruntime-gpu==1.13.1

RUN git clone https://github.com/isletennos/MMVC_Client.git
RUN cd MMVC_Client && git checkout 04f3fec4fd82dea6657026ec4e1cd80fb29a415c && cd -


WORKDIR /
ADD dummy /

RUN git clone  --depth 1 https://github.com/w-okada/voice-changer.git


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
COPY --from=base --chmod=777 /MMVC_Client /voice-changer/server/MMVC_Client
RUN chmod 0777 /voice-changer/server/MMVC_Client
RUN chmod 0777 /voice-changer/server

ADD /setup.sh  /voice-changer/server
ADD /exec.sh  /voice-changer/server


WORKDIR /voice-changer/server
ENTRYPOINT ["/bin/bash", "setup.sh"]
CMD [ "-h"]
