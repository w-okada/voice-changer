FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y emacs git curl python3-pip libsndfile1-dev ffmpeg

RUN git clone --depth 1 -b v1.5.0.0_SiFiGAN https://github.com/isletennos/MMVC_Trainer.git
ADD /model/D_v15_best.pth /MMVC_Trainer/fine_model/
ADD /model/G_v15_best.pth /MMVC_Trainer/fine_model/

RUN pip install pydub==0.25.1
RUN pip install tqdm==4.64.1
RUN pip install librosa==0.9.2
RUN pip install torch==1.13.1
RUN pip install torchaudio==0.13.1
RUN pip install psutil==5.9.4
RUN pip install tensorboard==2.11.2
RUN pip install pytz==2022.7.1
RUN pip install pyworld==0.3.2
RUN pip install retry==0.9.2
RUN pip install h5py==3.8.0
RUN pip install matplotlib==3.6.3

ADD /warmup.py /
RUN python3 warmup.py
WORKDIR /MMVC_Trainer/

RUN cp -r /MMVC_Trainer/configs /MMVC_Trainer/configs_org

# python3 normalize.py True
# python3 create_dataset.py -f train_config -s 24000 -m dataset/multi_speaker_correspondence.txt
# tensorboard --logdir logs --port 5000
# python3 train_ms.py -c configs/train_config.json -m 20220306_24000 -fg fine_model/G_v15_best.pth -fd fine_model/D_v15_best.pth

#$ docker run --gpus all --rm -ti -v `pwd`/trainer/dataset:/MMVC_Trainer/dataset -v `pwd`/trainer/configs:/MMVC_Trainer/configs -v `pwd`/trainer/F0:/MMVC_Trainer/F0 -v `pwd`/trainer/cF0:/MMVC_Trainer/cF0 -v `pwd`/trainer/units:/MMVC_Trainer/units -v `pwd`/trainer/logs:/MMVC_Trainer/logs -v `pwd`/trainer/filelists:/MMVC_Trainer/filelists -p 5000:5000 trainer /bin/bash





