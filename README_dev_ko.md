## 개발자용

[English](/README_dev_en.md) [Korean](/README_dev_ko.md)

## 전제

- Linux(ubuntu, debian) or WSL2, (다른 리눅스 배포판과 Mac에서는 테스트하지 않았습니다)
- Anaconda

## 준비

1. Anaconda 가상 환경을 작성한다

```
$ conda create -n vcclient-dev python=3.10
$ conda activate vcclient-dev
```
 
2. 리포지토리를 클론한다

```
$ git clone https://github.com/w-okada/voice-changer.git
```

## 서버 개발자용

1. 모듈을 설치한다

1-1. GPU가 없는 경우

```
$ python -m pip install -r server/requirements_cpuonly.txt
```

1-2. NVIDIA GPU를 사용하는 경우

`server/requirements_nvidia.txt` 파일에서 CUDA 버전에 따라 아래 `cu128` 부분을 적절히 수정하십시오.
기본 `cu128`은 CUDA 12.8용입니다.

```
--extra-index-url https://download.pytorch.org/whl/cu128

torch==2.7.0+cu128
torchaudio==2.7.0+cu128
```

다음을 실행하십시오:

```
$ python -m pip install -r server/requirements_nvidia.txt
```

2. 서버를 구동한다

다음 명령어로 구동합니다. 여러 가중치에 대한 경로는 환경에 맞게 변경하세요.

```
$ python3 MMVCServerSIO.py -p 18888 --https true \
    --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
    --content_vec_500_onnx pretrain/content_vec_500.onnx \
    --content_vec_500_onnx_on true \
    --hubert_base pretrain/hubert_base.pt \
    --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
    --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
    --nsf_hifigan pretrain/nsf_hifigan/model \
    --crepe_onnx_full pretrain/crepe_onnx_full.onnx \
    --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx \
    --rmvpe pretrain/rmvpe.pt \
    --model_dir model_dir \
    --samples samples.json
```

브라우저(Chrome에서만 지원)에서 접속하면 화면이 나옵니다.

2-1. 문제 해결법

(1) OSError: PortAudio library not found
다음과 같은 메시지가 나올 경우에는 추가 라이브러리를 설치해야 합니다.

```
OSError: PortAudio library not found
```

ubuntu(wsl2)인 경우에는 아래 명령어로 설치할 수 있습니다.

```
$ sudo apt-get install libportaudio2
$ sudo apt-get install libasound-dev
```

(2) 서버 구동이 안 되는데요?!

클라이언트는 자동으로 구동되지 않습니다. 브라우저를 실행하고 콘솔에 표시된 URL로 접속하세요.

(3) Could not load library libcudnn_cnn_infer.so.8
WSL를 사용 중이라면 `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory`라는 메시지가 나오는 경우가 있습니다.
잘못된 경로가 원인인 경우가 많습니다. 아래와 같이 경로를 바꾸고 실행해 보세요.
.bashrc 등 구동 스크립트에 추가해 두면 편리합니다.

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

- 참고
  - https://qiita.com/cacaoMath/items/811146342946cdde5b83
  - https://github.com/microsoft/WSL/issues/8587

3. 개발하세요

### Appendix

1. Win + Anaconda일 때 (not supported)

pytorch를 conda가 없으면 gpu를 인식하지 않을 수 있습니다.

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

또한 추가로 아래 내용도 필요합니다.

```
pip install chardet
pip install numpy==1.24.0
```

## 클라이언트 개발자용

1. 모듈을 설치하고 한번 빌드합니다

```
cd client
cd lib
npm install
npm run build:dev
cd ../demo
npm install
npm run build:dev
```

2. 개발하세요
