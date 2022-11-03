#!/bin/bash
set -eu

DOCKER_IMAGE=dannadori/voice-changer:20221104_062009
#DOCKER_IMAGE=voice-changer


MODE=$1
PARAMS=${@:2:($#-1)}

### DEFAULT VAR ###
DEFAULT_EX_PORT=18888
DEFAULT_USE_GPU=on # on|off
# DEFAULT_VERBOSE=off # on|off

### ENV VAR ###
EX_PORT=${EX_PORT:-${DEFAULT_EX_PORT}}
USE_GPU=${USE_GPU:-${DEFAULT_USE_GPU}}
# VERBOSE=${VERBOSE:-${DEFAULT_VERBOSE}}

#echo $EX_PORT $USE_GPU $VERBOSE

### INTERNAL SETTING ###
TENSORBOARD_PORT=6006
SIO_PORT=8080


### 
if [ "${MODE}" = "MMVC_TRAIN" ]; then
    echo "トレーニングを開始します"

    docker run -it --gpus all --shm-size=128M \
        -v `pwd`/exp/${name}/dataset:/MMVC_Trainer/dataset \
        -v `pwd`/exp/${name}/logs:/MMVC_Trainer/logs \
        -v `pwd`/exp/${name}/filelists:/MMVC_Trainer/filelists \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -e VERBOSE=${VERBOSE} \        
        -p ${EX_PORT}:6006 $DOCKER_IMAGE "$@"

elif [ "${MODE}" = "MMVC" ]; then
    if [ "${USE_GPU}" = "on" ]; then
        echo "MMVCを起動します(with gpu)"

        docker run -it --gpus all --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -p ${EX_PORT}:8080 $DOCKER_IMAGE "$@"
    else
        echo "MMVCを起動します(only cpu)"
        docker run -it --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -p ${EX_PORT}:8080 $DOCKER_IMAGE "$@"
    fi
else
    echo "
usage: 
    $0 <MODE> <params...>
    MODE: select one of ['MMVC_TRAIN', 'MMVC']
" >&2
fi


