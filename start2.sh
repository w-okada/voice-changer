#!/bin/bash
set -eu

DOCKER_IMAGE=dannadori/voice-changer:20230113_000254
# DOCKER_IMAGE=voice-changer

if [ $# = 0 ]; then
    echo "
    usage:
        $0 <MODE> <params...>
        MODE: select one of ['TRAIN', 'MMVC']
    " >&2
    exit 1
fi

MODE=$1

### DEFAULT VAR ###
DEFAULT_EX_PORT=18888
DEFAULT_EX_TB_PORT=16006
DEFAULT_USE_GPU=on # on|off
# DEFAULT_VERBOSE=off # on|off

### ENV VAR ###
EX_PORT=${EX_PORT:-${DEFAULT_EX_PORT}}
EX_TB_PORT=${EX_TB_PORT:-${DEFAULT_EX_TB_PORT}}
USE_GPU=${USE_GPU:-${DEFAULT_USE_GPU}}
# VERBOSE=${VERBOSE:-${DEFAULT_VERBOSE}}

#echo $EX_PORT $USE_GPU $VERBOSE


### 
if [ "${MODE}" = "TRAIN" ]; then
    echo "トレーニングを開始します"

    docker run -it --rm --gpus all --shm-size=128M \
        -v `pwd`/work_dir/logs:/voice-changer-internal/voice-change-service/MMVC_Trainer/logs \
        -v `pwd`/work_dir/dataset:/voice-changer-internal/voice-change-service/MMVC_Trainer/dataset \
        -v `pwd`/work_dir/info:/voice-changer-internal/voice-change-service/info \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_PORT=${EX_PORT} -e EX_TB_PORT=${EX_TB_PORT} \
        -e EX_IP="`hostname -I`" \
        -p ${EX_PORT}:8080 -p ${EX_TB_PORT}:6006 \
        $DOCKER_IMAGE "$@"


elif [ "${MODE}" = "MMVC" ]; then
    if [ "${USE_GPU}" = "on" ]; then
        echo "MMVCを起動します(with gpu)"

        docker run -it --rm --gpus all --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -p ${EX_PORT}:8080 \
        $DOCKER_IMAGE "$@"
    else
        echo "MMVCを起動します(only cpu)"
        docker run -it --rm --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -p ${EX_PORT}:8080 \
        $DOCKER_IMAGE "$@"
    fi
else
    echo "
usage: 
    $0 <MODE> <params...>
    MODE: select one of ['TRAIN', 'MMVC']
" >&2
fi


