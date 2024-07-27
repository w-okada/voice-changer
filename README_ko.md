## VC Client

[English](/README_en.md) [Japanese](/README.md)

## What's New!
- Beatrice V2 훈련 코드 공개!!!
  - [훈련 코드 리포지토리](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [Colab 버전](https://github.com/w-okada/beatrice-trainer-colab)
- v.2.0.47-alpha
  - 기능:
    - 추가 프레임 확장
  - 버그 수정:
    - Beatrice의 기본 화자 ID 변경
    - 모델 파일 이름이 너무 길 때의 오류 수정
    - 모니터 장치를 none으로 설정했을 때의 처리.
- v.2.0.45-alpha
  - bugfix
    - 음량 조절
  
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
