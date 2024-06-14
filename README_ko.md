## VC Client

[English](/README_en.md) [Japanese](/README.md)

## What's New!
- v.2.0.24-alpha
  - 버그 수정:
    - 모드 전환 시 소리가 나지 않는 문제를 해결
  - 기타:
    - 로거 강화
    - 에러 화면 강화
- v.2.0.23-alpha
  - 에디션 재정리
    - win_std: 일반적인 윈도우 사용자 대상. DirectML을 통한 하드웨어 가속이 ONNX 모델과 torch 모델 모두에서 가능합니다.
    - win_cuda: Nvidia GPU 소유자 대상. CUDA를 통한 하드웨어 가속이 ONNX 모델과 torch 모델 모두에서 가능합니다. CUDA 12.4 이상 필요.
    - mac: AppleSilicon(M1 등) 사용자 대상.
  - 기능
    - 클라이언트 모드에서 동작 시 출력 버퍼를 조정하는 기능 추가
  - 버그 수정:
    - RVC의 torch 모델을 onnx 모델로 내보낼 때 index와 icon을 유지하는 문제 수정
  - 기타:
    - 로거 강화

- v.2.0.20-alpha
  - Support for torch-cuda. See the edition description [here](docs/01_basic_v2.0.z.md).
  - Bugfix:
    - Unified file encoding to UTF-8
- v.2.0.16-alpha
  - torch-dml 실험적 버전을 지원. 에디션에 대한 설명은 [여기](docs/01_basic_v2.0.z.md)를 참조.
  - 버그 수정:
    - rvc 파일 업로드 시 pth와 index 파일을 동시에 업로드할 수 없는 문제를 해결.
  
- v.2.0.13-alpha
  - onnxruntime-gpu 지원 추가. CUDA 에디션 릴리스.
  - 버그 수정:
    - onnxcrepe 관련 문제 해결
    - Beatrice v2 API의 ID 선택 문제 수정
  - 기타:
    - 로거 강화
- v. 2.0.6-alpha
  - 신규:
    - M1 계열 Mac에 대응했습니다.
      - M1 MBA(monterey), M2 Pro MBP(venture)에서의 동작 실적이 있습니다.
      - sonoma에서의 보고를 기다리고 있습니다.
  - 버그 수정:
    - Beatrice의 스피커 선택 시 pitch가 원래대로 돌아가는 버그를 수정했습니다.
  - 기타:
    - 오류 분석을 위한 정보 획득 강화
  
- v.2.0.5-alpha
  - VCClient가 두 번째 버전으로 리부트 되었습니다.
  - 대폭적인 소프트웨어 구조 변경으로 확장 용이성을 높였습니다.
  - REST API를 제공하여 서드파티에서 클라이언트 개발을 용이하게 했습니다.
  - 에디션 체계를 새롭게 개편했습니다.
    - 스탠다드 에디션(win)은 GPU 유무와 상관없이 onnx 모델로 실행되는 것이 기본입니다. torch 모델은 onnx 모델로 변환한 후 사용하십시오. GPU를 소유한 사용자는 onnx 모델에서만 하드웨어 가속이 유효합니다.
    - cuda 에디션(win)은 Nvidia GPU에 특화된 튜닝이 되어 있습니다. 스탠다드 에디션에 비해 더욱 빠른 속도가 가능합니다. onnx 모델에서만 하드웨어 가속이 유효합니다.
    - torch 모델은 pytorch 모델도 하드웨어 가속을 지원합니다.
    - mac 에디션은 Apple Silicon을 탑재한 Mac 사용자들을 위한 것입니다.
    - linux 사용자나 python에 대한 지식이 있는 분들은 리포지토리를 클론하여 실행할 수도 있습니다.
  - 현재 Alpha 버전에서는 스탠다드 에디션만 제공됩니다.
  
# VC Client란
                                                                                                                                                     
1. 각종 음성 변환 AI(VC, Voice Conversion)를 활용해 실시간 음성 변환을 하기 위한 클라이언트 소프트웨어입니다. 지원하는 음성 변환 AI는 다음과 같습니다.
- 지원하는 음성 변환 AI (지원 VC)
  - [RVC(Retrieval-based-Voice-Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
  - [Beatrice JVS Corpus Edition](https://prj-beatrice.com/) * experimental,  (***NOT MIT Licnsence*** see [readme](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/)) *  Only for Windows, CPU dependent
  - 
1. 이 소프트웨어는 네트워크를 통한 사용도 가능하며, 게임 등 부하가 큰 애플리케이션과 동시에 사용할 경우 음성 변화 처리의 부하를 외부로 돌릴 수도 있습니다.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

1. 여러 플랫폼을 지원합니다.

- Windows, Mac(M1), Linux, Google Colab (MMVC만 지원)

1. REST API를 제공합니다.

- curl과 같은 OS에 내장된 HTTP 클라이언트를 사용하여 조작할 수 있습니다.
- 이를 통해 다음과 같은 것을 쉽게 실현할 수 있습니다.
  - 사용자가 .bat 등의 스크립트 파일로 REST API를 호출하는 처리를 바로가기로 등록한다.
  - 원격에서 조작할 수 있는 간이 클라이언트를 작성한다.
  - 등등.

# 다운로드
[Hugging Face](https://huggingface.co/wok000/vcclient000/tree/main)에서 다운로드하세요.

# 매뉴얼

[매뉴얼](docs/01_basic_v2.0.z.md)


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
