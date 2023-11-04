## 모니터링 튜토리얼

v.1.5.3.7부터 추가된 server device mode의 monitor output에 대한 설명입니다.

## v.1.5.3.6 이전의 구성

출력 대상 장치를 하나만 설정할 수 있었습니다. Discord나 Zoom 등 다른 애플리케이션에서 사용하기 위해서는 일반적으로 출력을 Voicemeeter와 같은 가상 오디오 장치 설정을 해야 할 필요가 있었습니다. 그로 인해 변환 후 음성을 확인하려면 가상 오디오 장치를 통해 확인해야 하는 등의 많은 수고가 필요했습니다(파란 화살표).

![image](https://github.com/w-okada/voice-changer/assets/48346627/faba8fdf-cfa5-468f-a56b-3fa986fb45a1)

## v.1.5.3.7 이후의 구성

v.1.5.3.7에서는 VC Client의 server device mode에서 출력 대상 장치를 하나 더 설정할 수 있게 됐습니다(빨간 화살표). 이를 통해 모니터링용으로 Voicemeeter를 거치지 않고 직접 wasapi 장치나 asio 장치로 출력할 수 있게 되어 지연이 적은 모니터링이 가능해졌습니다.

![image](https://github.com/w-okada/voice-changer/assets/48346627/1d5065eb-b042-4521-ade3-66828c87a712)

## 사용 방법

장치 설정 구역에서 server device mode를 선택하세요. 샘플링 레이트(S.R.), input, output, monitor를 설정할 수 있게 됩니다.

![image](https://github.com/w-okada/voice-changer/assets/48346627/c15e6800-75ec-410b-87f2-c96d0c697c91)

## 주의 사항

server device mode에서 사용하는 input, output, monitor 각 장치의 샘플링 레이트는 일치해야 합니다. 일치하지 않을 경우에는 콘솔에 자세한 정보가 표시되므로 GUI에서 각 장치가 지원하는 샘플링 레이트를 지정하세요.

### 예시

![image](https://github.com/w-okada/voice-changer/assets/48346627/d621d356-5710-4766-932e-43b7d520df5f)

샘플링 레이트가 일치하지 않으면 위와 같이 표시됩니다.

(1)는 현재 GUI에서 장치에 지정된 샘플링 레이트 지원 여부를 표시합니다. False인 장치는 지원하지 않습니다.

(2)에서 각 장치에서 지원하는 샘플링 레이트를 표시합니다. input, output, monitor 전부 지원하는 샘플링 레이트를 지정하세요. 예시에서는 48000으로 지정했습니다.

## 팁

### 첫 번째

사용 환경에 따라 크게 달라지겠지만, 개발자 환경에서는 input, monitor를 wasapi 장치로 output을 임의로 설정해 상당히 낮은 지연으로 사용할 수 있었습니다.
(RTX 4090 사용)

### 두 번째

Wasapi의 샘플링 레이트는 장치에서 설정한 것만 선택할 수 있습니다. 이 설정은 Windows 사운드 설정에서 변경할 수 있습니다.(Win11)

![image](https://github.com/w-okada/voice-changer/assets/48346627/300c8cf0-cb7d-4f24-8253-fa313caee5df)
