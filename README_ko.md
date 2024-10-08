## VC Client

[English](/README_en.md) [Korean](/README_ko.md)

## 새로운 기능!
- 자매품으로 텍스트 음성 변환 클라이언트를 출시하였습니다.
  - 간단한 인터페이스로 음성 생성을 즐길 수 있습니다.
  - 자세한 내용은 [여기](https://github.com/w-okada/ttsclient)를 참조하세요.
- Beatrice V2 훈련 코드 공개!!!
  - [훈련 코드 리포지토리](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [Colab 버전](https://github.com/w-okada/beatrice-trainer-colab)
- v.2.0.65-beta
  - [여기를 참조하십시오](https://github.com/w-okada/voice-changer/tree/v.2)
  - new feature: Beatrice v2 beta.2를 지원하여 더 높은 품질의 음성 변환이 가능해졌습니다
- v.2.0.61-alpha
  - [여기를 참조하십시오](https://github.com/w-okada/voice-changer/tree/v.2)
  - 기능:
    - 크로스페이드 시간을 지정할 수 있게 되었습니다.
  - 버그 수정:
    - 모델 병합 시 사용하지 않는 모델의 요소를 0으로 설정해도 동작하도록 수정되었습니다.
- v.2.0.60-alpha
  - [여기를 참조하십시오](https://github.com/w-okada/voice-changer/tree/v.2)
  - feature:
    - [darkmode](https://github.com/w-okada/voice-changer/issues/1306)
    - [re-introduce pytorch rmvpe](https://github.com/w-okada/voice-changer/issues/1319)
    - [wasapi 독점 모드 선택됨](https://github.com/w-okada/voice-changer/issues/1305)
- v.2.0.58-alpha
  - [여기를 참조하십시오](https://github.com/w-okada/voice-changer/tree/v.2)
  - 기능:
    - SIO 브로드캐스팅
    - ngrok 내장 (실험적)
  - 향상된 점:
    - 모바일 폰을 위한 튜닝.
  - 버그 수정:
    - macOS에서 CUI 메시지 글자 깨짐 문제
- v.2.0.55-alpha
  - [여기를 참조하십시오](https://github.com/w-okada/voice-changer/tree/v.2)
  - 개선:
    - RVC의 CPU 부하 감소
    - WebSocket 지원
  - 변경:
    - 시작 배치에서 no_cui 옵션 활성화

# VC Client란
                                                                                                                                                     
1. 각종 음성 변환 AI(VC, Voice Conversion)를 활용해 실시간 음성 변환을 하기 위한 클라이언트 소프트웨어입니다. 지원하는 음성 변환 AI는 다음과 같습니다.

- 지원하는 음성 변환 AI (지원 VC)
  - [MMVC](https://github.com/isletennos/MMVC_Trainer) (only v1)
  - [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) (only v1)
  - [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
  - [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) (only v1)
  - [Beatrice JVS Corpus Edition](https://prj-beatrice.com/) * experimental,  (***NOT MIT License*** see [readme](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/)) *  Only for Windows, CPU dependent (only v1)
  - [Beatrice v2](https://prj-beatrice.com/) (only for v2)
  - 
1. 이 소프트웨어는 네트워크를 통한 사용도 가능하며, 게임 등 부하가 큰 애플리케이션과 동시에 사용할 경우 음성 변화 처리의 부하를 외부로 돌릴 수도 있습니다.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

3. 여러 플랫폼을 지원합니다.

- Windows, Mac(M1), Linux, Google Colab (MMVC만 지원)
## 관련 소프트웨어
- [실시간 음성 변조기 VCClient](https://github.com/w-okada/voice-changer)
- [텍스트 읽기 소프트웨어 TTSClient](https://github.com/w-okada/ttsclient)
- [실시간 음성 인식 소프트웨어 ASRClient](https://github.com/w-okada/asrclient)
# 사용 방법

크게 두 가지 방법으로 사용할 수 있습니다. 난이도 순서는 다음과 같습니다.

- 사전 빌드된 Binary 사용
- Docker, Anaconda 등으로 구축된 개발 환경에서 사용

이 소프트웨어나 MMVC에 익숙하지 않은 분들은 위에서부터 차근차근 익숙해지길 추천합니다.

## (1) 사전 빌드된 Binary(파일) 사용

- 실행 형식 바이너리를 다운로드하여 실행할 수 있습니다.

- 튜토리얼은 [이곳](tutorials/tutorial_rvc_ko_latest.md)을 확인하세요。([네트워크 문제 해결법](https://github.com/w-okada/voice-changer/blob/master/tutorials/trouble_shoot_communication_ko.md))

- [Google Colaboratory](https://github.com/w-okada/voice-changer/tree/v.2/w_okada's_Voice_Changer_version_2_x.ipynb) で簡単にお試しいただけるようになりました。左上の Open in Colab のボタンから起動できます。

<img src="https://github.com/w-okada/voice-changer/assets/48346627/3f092e2d-6834-42f6-bbfd-7d389111604e" width="400" height="150">

- Windows 버전과 Mac 버전을 제공하고 있습니다. [Hugging Face](https://huggingface.co/wok000/vcclient000/tree/main)에서 다운로드할 수 있습니다.
- Windows용 v2
  - `vcclient_win_std_xxx.zip`를 다운로드하여 사용하세요. GPU를 사용하지 않고도 (어느 정도 고성능의) CPU를 사용한 음성 변환이나, DirectML을 사용해 GPU(AMD, Nvidia)를 활용한 음성 변환이 가능합니다. v2에서는 torch와 onnx 모두를 지원합니다.
  - Nvidia GPU를 가지고 계신 분들은 `vcclient_win_cuda_xxx.zip`를 사용하시면 더 빠른 음성 변환이 가능합니다.
- Mac (Apple Silicon)용 v2
  - `vcclient_mac_xxx.zip`를 다운로드하여 사용하세요.
- v1
  - Windows와 NVIDIA GPU를 사용하는 분은 ONNX(cpu, cuda), PyTorch(cpu, cuda)를 다운로드하세요.
  - Windows와 AMD/Intel GPU를 사용하는 분은 ONNX(cpu, DirectML), PyTorch(cpu, cuda)를 다운로드하세요 AMD/Intel GPU는 ONNX 모델을 사용할 때만 적용됩니다.
  - 그 외 GPU도 PyTorch, Onnxruntime가 지원할 경우에만 적용됩니다.
  - Windows에서 GPU를 사용하지 않는 분은 ONNX(cpu, cuda), PyTorch(cpu, cuda)를 다운로드하세요.

- Windows 버전은 다운로드한 zip 파일의 압축을 풀고 `start_http.bat`를 실행하세요.

- Mac 버전은 다운로드한 파일을 풀고 `startHttp.command`를 실행하세요. 확인되지 않은 개발자 메시지가 나오면 다시 control 키를 누르고 클릭해 실행하세요(or 오른쪽 클릭으로 실행하세요).

- 처음 실행할 때는 인터넷으로 여러 데이터를 다운로드합니다. 다운로드할 때 시간이 좀 걸릴 수 있습니다. 다운로드가 완료되면 브라우저가 실행됩니다.

- 원격으로 접속할 때는 http 대신 https `.bat` 파일(win)、`.command` 파일(mac)을 실행하세요.

- DDPS-SVC의 encoder는 hubert-soft만 지원합니다.


## (2) Docker나 Anaconda 등으로 구축된 개발 환경에서 사용

이 리포지토리를 클론해 사용할 수 있습니다. Windows에서는 WSL2 환경 구축이 필수입니다. 또한, WSL2 상에 Docker나 Anaconda 등의 가상환경 구축이 필요합니다. Mac에서는 Anaconda 등의 Python 가상환경 구축이 필요합니다. 사전 준비가 필요하지만, 많은 환경에서 이 방법이 가장 빠르게 작동합니다. **<font color="red"> GPU가 없어도 나름 최근 출시된 CPU가 있다면 충분히 작동할 가능성이 있습니다</font>(아래 실시간성 항목 참조)**.

[WSL2와 Docker 설치 설명 영상](https://youtu.be/POo_Cg0eFMU)

[WSL2와 Anaconda 설치 설명 영상](https://youtu.be/fba9Zhsukqw)

Docker에서 실행은 [Docker를 사용](docker_vcclient/README_ko.md)을 참고해 서버를 구동하세요.

Anaconda 가상 환경에서 실행은 [서버 개발자용 문서](README_dev_ko.md)를 참고해 서버를 구동하세요.

# 문제 해결법

- [통신편](tutorials/trouble_shoot_communication_ko.md)


# 개발자 서명에 대하여

이 소프트웨어는 개발자 서명이 없습니다. 本ソフトウェアは開発元の署名しておりません。下記のように警告が出ますが、コントロールキーを押しながらアイコンをクリックすると実行できるようになります。これは Apple のセキュリティポリシーによるものです。実行は自己責任となります。

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)
(이미지 번역: ctrl을 누른 채로 클릭)

# 감사의 말

- [立ちずんだもん素材](https://seiga.nicovideo.jp/seiga/im10792934)
- [いらすとや](https://www.irasutoya.com/)
- [つくよみちゃん](https://tyc.rei-yumesaki.net/)

```
  이 소프트웨어의 음성 합성에는 무료 소재 캐릭터 「つくよみちゃん(츠쿠요미 짱)」이 무료 공개하고 있는 음성 데이터를 사용했습니다.■츠쿠요미 짱 말뭉치(CV.夢前黎)
  https://tyc.rei-yumesaki.net/material/corpus/
  © Rei Yumesaki
```

- [あみたろの声素材工房](https://amitaro.net/)
- [れぷりかどーる](https://kikyohiroto1227.wixsite.com/kikoto-utau)

# 이용약관

- 실시간 음성 변환기 츠쿠요미 짱은 츠쿠요미 짱 말뭉치 이용약관에 따라 다음과 같은 목적으로 변환 후 음성을 사용하는 것을 금지합니다.

```

■사람을 비판·공격하는 행위. ("비판·공격"의 정의는 츠쿠요미 짱 캐릭터 라이센스에 준합니다)

■특정 정치적 입장·종교·사상에 대한 찬반을 논하는 행위.

■자극적인 표현물을 무분별하게 공개하는 행위.

■타인에게 2차 창작(소재로서의 활용)을 허가하는 형태로 공개하는 행위.
※감상용 작품으로서 배포·판매하는 건 문제없습니다.
```

- 실시간 음성 변환기 아미타로는 あみたろの声素材工房(아미타로의 음성 소재 공방)의 다음 이용약관에 따릅니다. 자세한 내용은 [이곳](https://amitaro.net/voice/faq/#index_id6)에 있습니다.

```
아미타로의 음성 소재나 말뭉치 음성으로 음성 모델을 만들거나, 음성 변환기나 말투 변환기 등을 사용해 본인 목소리를 아미타로의 목소리로 변환해 사용하는 것도 괜찮습니다.

단, 그 경우에는 반드시 아미타로(혹은 코하루네 아미)의 음성으로 변환한 것을 명시하고, 아미타로(및 코하루네 아미)가 말하는 것이 아님을 누구나 알 수 있도록 하십시오.
또한 아미타로의 음성으로 말하는 내용은 음성 소재 이용약관의 범위 내에서만 사용해야 하며, 민감한 발언은 삼가십시오.
```

- 실시간 음성 변환기 키코토 마히로는 れぷりかどーる(레플리카 돌)의 이용약관에 따릅니다. 자세한 내용은 [이곳](https://kikyohiroto1227.wixsite.com/kikoto-utau/ter%EF%BD%8Ds-of-service)에 있습니다.

# 면책 사항

이 소프트웨어의 사용 또는 사용 불능으로 인해 발생한 직접 손해·간접 손해·파생적 손해·결과적 손해 또는 특별 손해에 대해 모든 책임을 지지 않습니다.

# (1) 레코더(트레이닝용 음성 녹음 앱)

MMVC 트레이닝용 음성을 간단하게 녹음할 수 있는 앱입니다.
Github Pages에서 실행할 수 있어서 브라우저만 있으면 다양한 플랫폼에서 사용할 수 있습니다.
녹음한 데이터는 브라우저에 저장됩니다. 외부로 유출되지 않습니다.

[녹음 앱 on Github Pages](https://w-okada.github.io/voice-changer/)

[설명 영상](https://youtu.be/s_GirFEGvaA)

# 이전 버전

| Version    | OS  | 프레임워크                        | link                                                                                           | 지원 VC                                                                       | 파일 크기 |
| ---------- | --- | --------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------- |
| v.1.5.2.9e | mac | ONNX(cpu), PyTorch(cpu,mps)       | [normal](https://drive.google.com/uc?id=1W0d7I7619PcO7kjb1SPXp6MmH5Unvd78&export=download) \*1 | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0, RVC                              | 796MB     |
|            | win | ONNX(cpu,cuda), PyTorch(cpu,cuda) | [normal](https://drive.google.com/uc?id=1tmTMJRRggS2Sb4goU-eHlRvUBR88RZDl&export=download) \*1 | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0, so-vits-svc 4.0v2, RVC, DDSP-SVC | 2872MB    |
| v.1.5.3.1  | mac | ONNX(cpu), PyTorch(cpu,mps)       | [normal](https://drive.google.com/uc?id=1oswF72q_cQQeXhIn6W275qLnoBAmcrR_&export=download) \*1 | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0, RVC                              | 796MB     |
|            | win | ONNX(cpu,cuda), PyTorch(cpu,cuda) | [normal](https://drive.google.com/uc?id=1AWjDhW4w2Uljp1-9P8YUJBZsIlnhkJX2&export=download) \*1 | MMVC v.1.5.x, MMVC v.1.3.x, so-vits-svc 4.0, so-vits-svc 4.0v2, RVC, DDSP-SVC | 2872MB    |

# For Contributor

이 리포지토리는 [CLA](https://raw.githubusercontent.com/w-okada/voice-changer/master/LICENSE-CLA)를 설정했습니다.
