#!/bin/bash
set -eu

DOCKER_IMAGE=dannadori/vcclient:20230627_072633
#DOCKER_IMAGE=vcclient

### DEFAULT VAR ###
DEFAULT_EX_PORT=18888
DEFAULT_USE_GPU=on # on|off
DEFAULT_USE_LOCAL=off # on|off
### ENV VAR ###
EX_PORT=${EX_PORT:-${DEFAULT_EX_PORT}}
USE_GPU=${USE_GPU:-${DEFAULT_USE_GPU}}
USE_LOCAL=${USE_LOCAL:-${DEFAULT_USE_LOCAL}}

if [ "${USE_LOCAL}" = "on" ]; then
    DOCKER_IMAGE=vcclient
fi

if [ "${USE_GPU}" = "on" ]; then
    echo "VC Client start...(with gpu)"
    docker run -it --rm --gpus all --shm-size=1024M \
    -e EX_IP="`hostname -I`" \
    -e EX_PORT=${EX_PORT} \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -p ${EX_PORT}:18888 \
    $DOCKER_IMAGE -p 18888 --https true \
        --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
        --content_vec_500_onnx pretrain/content_vec_500.onnx \
        --content_vec_500_onnx_on true \
        --hubert_base pretrain/hubert_base.pt \
        --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
        --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
        --nsf_hifigan pretrain/nsf_hifigan/model \
        --crepe_onnx_full pretrain/crepe_onnx_full.onnx \
        --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx \
        --model_dir model_dir \
        --samples samples.json
else
    echo "VC Client start...(cpu)"
    docker run -it --rm --shm-size=1024M \
    -e EX_IP="`hostname -I`" \
    -e EX_PORT=${EX_PORT} \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    -p ${EX_PORT}:18888 \
    $DOCKER_IMAGE -p 18888 --https true \
        --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
        --content_vec_500_onnx pretrain/content_vec_500.onnx \
        --content_vec_500_onnx_on true \
        --hubert_base pretrain/hubert_base.pt \
        --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
        --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
        --nsf_hifigan pretrain/nsf_hifigan/model \
        --crepe_onnx_full pretrain/crepe_onnx_full.onnx \
        --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx \
        --model_dir model_dir \
        --samples samples.json
fi


