## 문제 해결법 통신편

음성이 전혀 변환되지 않는 경우나 변환 후 음성이 이상하게 될 경우에는 음성 변환 과정에서 문제점을 찾아야 합니다.

이 문서에서는 어떤 부분에서 문제가 발생하는지 대략적으로 찾을 수 있는 방법에 대한 설명입니다.

## VC Client의 구성과 문제 구분

<img src="https://user-images.githubusercontent.com/48346627/235551041-6eed4035-5542-47d1-bbd3-31fa7842011b.png" width="720">

사용자(음성 입력) → GUI → 변환 전 음성(1) → 서버 → 변환 후의 음성(2) → GUI에 도착한 변환 후 음성(3) → 스피커(음성 출력)

VC Client는 이미지 자료처럼 GUI(클라이언트)가 마이크를 통해 음성을 받고, 서버에서 변환하는 구성을 하고 있습니다.

VC Client는 이미지 자료 음성이 세 곳에서 어떤 상태인지 확인할 수 있습니다.
정상 상태로 음성이 녹음됐다면 이 과정까지는 처리가 잘 된 것이고, 이후부터 문제를 찾으면 됩니다(문제 구분이라고 합니다).

## 음성의 상태 확인 방법

### (1)(2)로 음성 상태 확인

<img src="https://github.com/w-okada/voice-changer/assets/48346627/f4845f1d-2e1a-49c1-a226-0e50be807f2d" width="720">

Analyzer의 Sampling을 시작한 상태에서 음성 변환을 시도해 보세요. 어느 정도 음성을 입력 후에 Samplling을 정지하면 in/out에 재생 버튼이 표시됩니다.

- in에는 'VC Client의 구성과 문제 구분'의 이미지 자료(1)(GUI→서버)의 음성이 녹음되어 있습니다. 마이크로 입력된 음성이 그대로 서버에 녹음될 테니 사용자의 음성이 녹음됐다면 정상입니다.
- out에는 'VC Client의 구성과 문제 구분'의 이미지 자료(2)(서버에서 변환한 후의 음성)의 음성이 녹음되어 있습니다. AI를 통해 변환된 음성이 녹음되어 있을 겁니다.

### (3)으로 음성 상태 확인

<img src="https://github.com/w-okada/voice-changer/assets/48346627/18ddfc2c-beb2-4e7a-8a06-1e00cc6ddb72" width="720">

Audio Output의 output record를 시작한 상태로 음성 변환을 시도해 보세요. 어느 정도 음성을 입력한 후에 정리하면 .wav 파일이 다운로드됩니다. 이 .wav 파일은 서버에서 전송된 변환 후의 음성이 녹음되어있을 겁니다.

## 음성 상태 확인 후

앞서 설명한 이미지 자료의 (1)~(3) 중에서 예상한 상태의 녹음 음성이 어디까지 진행됐나 파악했다면, 예상한 상태의 음성이 녹음된 곳 이후에도 문제가 없는지 검토하세요.

### (1)에서의 음성 상태가 이상한 경우

#### 음성 파일로 확인

음성 파일로 변환이 되는지 확인하세요.

예를 들어, 다음 파일을 사용해 보세요.

- [sample_jvs001](https://drive.google.com/file/d/142aj-qFJOhoteWKqgRzvNoq02JbZIsaG/view) from [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [sample_jvs001](https://drive.google.com/file/d/1iCErRzCt5-6ftALcic9w5zXWrzVXryIA/view) from [JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)

#### 마이크 입력 확인

마이크 입력 자체에 문제가 있을 가능성이 있습니다. 녹음 프로그램 등을 사용해 마이크 입력을 확인하세요.
또한 [이 녹음 사이트](https://w-okada.github.io/voice-changer/)는 VC Client의 자매품으로 마이크 입력 처리가 거의 동일하게 이루어져 참고할 만합니다. (설치 필요 없음. 브라우저에서만 동작합니다.)
