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

WORKDIR /
ADD dummy /

RUN git clone https://github.com/w-okada/voice-changer.git -b v.1.5.3.5a



RUN chmod 0777 /voice-changer/server
WORKDIR /voice-changer/server
RUN pip install -r requirements.txt

RUN git clone https://github.com/isletennos/MMVC_Client.git MMVC_Client_v13
RUN cd MMVC_Client_v13 && git checkout 04f3fec4fd82dea6657026ec4e1cd80fb29a415c && cd -

RUN git clone https://github.com/isletennos/MMVC_Client.git MMVC_Client_v15
RUN cd MMVC_Client_v15 && git checkout 461cb231b57cbb17243110eaac8435d9cca24a26 && cd -

RUN git clone https://github.com/StarStringStudio/so-vits-svc.git so-vits-svc-40
RUN cd so-vits-svc-40 && git checkout 293fe0bc03a583df17846c715c179152745c1b8d && cd -
RUN git clone https://github.com/StarStringStudio/so-vits-svc.git so-vits-svc-40v2
RUN cd so-vits-svc-40v2 && git checkout 08c70ff3d2f7958820b715db2a2180f4b7f92f8d && cd -
RUN git clone https://github.com/yxlllc/DDSP-SVC.git DDSP-SVC
RUN cd DDSP-SVC && git checkout e716efb14aee747492bc6ac1deb07506c26ed550 && cd -
RUN git clone https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI.git RVC
RUN cd RVC && git checkout 563c64ded9925085f547b13ecafb8eb9c182952b && cd -

ADD /setup.sh  /voice-changer/server
ADD /exec.sh  /voice-changer/server


ENTRYPOINT ["/bin/bash", "setup.sh"]
