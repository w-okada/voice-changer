FROM dannadori/voice-changer-internal:20221112_114121 as front
FROM debian:bullseye-slim as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y python3-pip git
RUN apt-get install -y espeak
RUN apt-get install -y cmake

#RUN git clone --depth 1 https://github.com/isletennos/MMVC_Trainer.git -b v1.3.1.3

# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


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
RUN pip install pyOpenSSL==22.0.0

RUN pip install pyopenjtalk==0.2.0
RUN pip install tensorboard==2.10.0

# WORKDIR /MMVC_Trainer/monotonic_align
# RUN cythonize -3 -i core.pyx \
#  && mv core.cpython-39-x86_64-linux-gnu.so monotonic_align/


FROM debian:bullseye-slim
RUN apt-get update \
        && apt-get install -y python3-pip espeak gosu libsndfile1-dev\
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

### Copy local resource
COPY fine_model/G_180000.pth /MMVC_Trainer/fine_model/G_180000.pth
COPY fine_model/D_180000.pth /MMVC_Trainer/fine_model/D_180000.pth

### Copy from base
COPY --from=base --chmod=777 /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages

### Copy from frontend
##### MMVC Trainer
COPY --from=front --chmod=777 /MMVC_Trainer /MMVC_Trainer
RUN chmod 0777 /MMVC_Trainer 

ADD /setup.sh  /MMVC_Trainer/
ADD /exec.sh  /MMVC_Trainer/

##### Voice changer Internal
COPY --from=front --chmod=777 /voice-changer-internal/frontend/dist /voice-changer-internal/frontend/dist
COPY --from=front --chmod=777 /voice-changer-internal/voice-change-service /voice-changer-internal/voice-change-service
RUN chmod 0777 /voice-changer-internal/voice-change-service 


## 歴史的な経緯でsetup.shをMMVC_Trainerに置いているのでそこを起動時のWORKDIRにしている。
WORKDIR /MMVC_Trainer

# ##### Soft VC
# COPY --from=front /hubert /hubert
# COPY --from=front /acoustic-model /acoustic-model
# COPY --from=front /hifigan /hifigan
# COPY --from=front /models /models 


ENTRYPOINT ["/bin/bash", "setup.sh"]
CMD [ "-h"]
