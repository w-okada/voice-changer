## Device Mode 튜토리얼

Device Mode에 대한 설명입니다.

[설명 영상](https://youtu.be/SUnRGCJ92K8?t=99)

## v.1.5.2.9 이전의 구성(client device mode)

v.1.5.2.9 이전에는 브라우저가 제어하는 마이크와 스피커를 사용해 음성 변환을 진행했습니다.
이것을 client device mode라 부릅니다(빨간 화살표).

![image](https://github.com/w-okada/voice-changer/assets/48346627/56c0766c-45c1-4b3d-af66-73443c232807)

## v.1.5.2.9 이후의 구성(client device mode / server device mode)

v.1.5.2.9부터 PC에 접속된 마이크와 스피커를 직접 VC Client에서 제어해 음성 변환을 진행하는 모드를 추가했습니다. 이것을 server device mode라 부릅니다(파란 화살표)。

![image](https://github.com/w-okada/voice-changer/assets/48346627/34c92e36-0662-4eeb-aac5-30cd1f4a5cd8)

## client device mode / server device mode의 장점과 단점

v.1.5.2.9 이후에는 client device mode와 server device mode 중에서 사용할 것을 선택할 수 있게 됐습니다.

- client device mode
  - 장점
    1. Chrome이 마이크/스피커의 어려운 처리를 대신해 준다.
    2. 잡음 제거 등의 Chrome이 가진 Web 회의 기능을 사용할 수 있다.
  - 단점
    1. 다소 지연이 발생할 수 있다.
- server device mode
  - 장점
    1. VC Client가 직접 마이크/스피커를 다뤄서 지연이 적다.
  - 단점
    1. 다룰 수 없는 마이크/스피커가 있을 수 있다.
    2. 잡음 제거 등 Chrome의 편리한 기능을 사용할 수 없다.

![image](https://github.com/w-okada/voice-changer/assets/48346627/fef1ee63-e853-4867-b4c8-bf0121495bb6)

사용자는 각 장점·단점을 고려해 구분하여 사용할 수 있습니다.