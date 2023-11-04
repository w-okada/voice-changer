## VC Client for Docker

[Japanese](./README.md) [English](./README_en.md)

## 빌드

리포지토리 폴더의 최상위 위치에서

```
npm run build:docker:vcclient
```

## 실행

리포지토리 폴더의 최상위 위치에서

```
bash start_docker.sh
```

브라우저(Chrome에서만 지원)로 접속하면 화면이 나옵니다.

## RUN with options

GPU를 사용하지 않는 경우에는

```
USE_GPU=off bash start_docker.sh
```

포트 번호를 변경하고 싶은 경우에는

```
EX_PORT=<port> bash start_docker.sh
```

로컬 이미지를 사용하고 싶은 경우에는

```
USE_LOCAL=on bash start_docker.sh
```

## Push to Repo (only for devs)

```
npm run push:docker:vcclient
```
