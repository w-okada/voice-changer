## VC Client

[English](/README_en.md) [Japanese](/README.md)

## What's New!
- Beatrice V2 훈련 코드 공개!!!
  - [훈련 코드 리포지토리](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [Colab 버전](https://github.com/w-okada/beatrice-trainer-colab)
- v.2.0.70-beta (only for m1 mac)
  - new feature:
    - M1 Mac 버전 VCClient에서도 Beatrice v2 beta.1을 지원합니다.
- v.2.0.69-beta (only for win)
  - 버그 수정:
    - 일부 예외 발생 시 시작 버튼이 표시되지 않는 버그를 수정
    - 서버 디바이스 모드의 출력 버퍼 조정
    - 서버 디바이스 모드 사용 중 설정 변경 시 샘플링 레이트가 변하는 버그 수정
    - 일본어 hubert 사용 시 버그 수정
  - 기타:
    - 서버 디바이스 모드에 호스트 API 필터 추가 (강조 표시)
- v.2.0.65-beta
  - new feature: Beatrice v2 beta.1를 지원하여 더 높은 품질의 음성 변환이 가능해졌습니다
- v.2.0.61-alpha
  - 기능:
    - 크로스페이드 시간을 지정할 수 있게 되었습니다.
  - 버그 수정:
    - 모델 병합 시 사용하지 않는 모델의 요소를 0으로 설정해도 동작하도록 수정되었습니다.
- v.2.0.58-alpha
  - 기능:
    - SIO 브로드캐스팅
    - ngrok 내장 (실험적)
  - 향상된 점:
    - 모바일 폰을 위한 튜닝.
  - 버그 수정:
    - macOS에서 CUI 메시지 글자 깨짐 문제
- v.2.0.55-alpha
  - 개선:
    - RVC의 CPU 부하 감소
    - WebSocket 지원
  - 변경:
    - 시작 배치에서 no_cui 옵션 활성화
  
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

## 관련 소프트웨어
- [실시간 음성 변조기 VCClient](https://github.com/w-okada/voice-changer)
- [텍스트 읽기 소프트웨어 TTSClient](https://github.com/w-okada/ttsclient)
- [실시간 음성 인식 소프트웨어 ASRClient](https://github.com/w-okada/asrclient)
- 
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
