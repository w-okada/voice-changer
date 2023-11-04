## VC Client for Docker

[Japanese](./README.md) 
[Korean](./README.md)

## Build

In root folder of repos.

```
npm run build:docker:vcclient
```

## preparation

Store weights of external models in `docker_vcclient/weights`. Which weights should be in the folder depends on which kind of VC you use.

```
$ tree docker_vcclient/weights/
docker_vcclient/weights/
├── checkpoint_best_legacy_500.onnx
├── checkpoint_best_legacy_500.pt
├── hubert-soft-0d54a1f4.pt
├── hubert_base.pt
└── nsf_hifigan
    ├── NOTICE.txt
    ├── NOTICE.zh-CN.txt
    ├── config.json
    └── model
```

## Run

In root folder of repos.

```
bash start_docker.sh
```

Access with Browser (currently only chrome is supported), then you can see gui.

## RUN with options

Without GPU

```
USE_GPU=off bash start_docker.sh
```

Specify port num

```
EX_PORT=<port> bash start_docker.sh
```

Use Local Image

```
USE_LOCAL=on bash start_docker.sh
```

## Push to Repo (only for devs)

```
npm run push:docker:vcclient
```
