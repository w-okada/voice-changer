FROM debian:bullseye-slim 

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    git \
    wget \
    emacs \
    protobuf-compiler \
    cmake \
    libsndfile1-dev


RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install Cython==0.29.32 pip install librosa==0.9.2 \
&& pip install librosa==0.9.2 \
&& pip install Unidecode==1.3.4 \
&& pip install phonemizer==3.2.1 \
&& pip install onnx \
&& pip install onnxsim \
&& pip install onnxruntime \
&& pip install onnx-simplifier \

RUN pip install --upgrade protobuf

RUN git clone https://github.com/isletennos/MMVC_Trainer.git -b v1.3.2.1

WORKDIR /MMVC_Trainer/monotonic_align
RUN cythonize -3 -i core.pyx \
 && mv core.cpython-39-x86_64-linux-gnu.so monotonic_align/

WORKDIR /MMVC_Trainer
