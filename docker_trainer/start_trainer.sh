#!/bin/bash
set -eu

DOCKER_IMAGE=dannadori/trainer:20230131_182050

docker run --gpus all --rm -ti \
    -v `pwd`/trainer/dataset:/MMVC_Trainer/dataset \
    -v `pwd`/trainer/configs:/MMVC_Trainer/configs \
    -v `pwd`/trainer/F0:/MMVC_Trainer/F0 \
    -v `pwd`/trainer/cF0:/MMVC_Trainer/cF0 \
    -v `pwd`/trainer/units:/MMVC_Trainer/units \
    -v `pwd`/trainer/logs:/MMVC_Trainer/logs \
    -v `pwd`/trainer/filelists:/MMVC_Trainer/filelists \
    -p 5000:5000 \
    $DOCKER_IMAGE /bin/bash




