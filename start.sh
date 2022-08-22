#!/bin/bash

EXP_NAME=$1
shift 

docker run -it --gpus all --shm-size=2g \
  -v `pwd`/exp/${EXP_NAME}/dataset:/MMVC_Trainer/dataset \
  -v `pwd`/exp/${EXP_NAME}/logs:/MMVC_Trainer/logs \
  -v `pwd`/exp/${EXP_NAME}/filelists:/MMVC_Trainer/filelists \
  -v `pwd`/vc_resources:/resources \
  -e LOCAL_UID=$(id -u $USER) \
  -e LOCAL_GID=$(id -g $USER) \
  -p 6008:6006 -p 8081:8080 mmvc_trainer_docker "$@"


