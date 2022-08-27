#!/bin/bash

# 参考:https://programwiz.org/2022/03/22/how-to-write-shell-script-for-option-parsing/

DOCKER_IMAGE=dannadori/voice-changer:20220827_105904
TENSORBOARD_PORT=6006
VOICE_CHANGER_PORT=8080

set -eu

echo "------"
echo "$@"
echo "------"

usage() {
    echo "
usage: 
    For training
        $0 [-t] -n <exp_name> [-b batch_size] [-r] 
            -t: トレーニングモードで実行する場合に指定してください。(train)
            -n: トレーニングの名前です。(name)
            -b: バッチサイズです。(batchsize)
            -r: トレーニング再開の場合に指定してください。(resume)
    For changing voice
        $0 [-v] [-c config] [-m model] [-g on/off]
            -v: ボイスチェンジャーモードで実行する場合に指定してください。(voice changer)
            -c: トレーニングで使用したConfigのファイル名です。(config)
            -m: トレーニング済みのモデルのファイル名です。(model)
            -g: GPU使用/不使用。デフォルトはonなのでGPUを使う場合は指定不要。(gpu)
    For help
        $0 [-h]
            -h: show this help
" >&2
}
warn () {
    echo "! ! ! $1 ! ! !"
    exit 1
}


training_flag=false
name=999_exp
batch_size=10
resume_flag=false

voice_change_flag=false
config=
model=
gpu=on

escape_flag=false

# オプション解析
while getopts tn:b:rvc:m:g:hx OPT; do
    case $OPT in
    t) 
        training_flag=true
        ;;
    n) 
        name="$OPTARG"
        ;;
    b) 
        batch_size="$OPTARG"
        ;;
    r) 
        resume_flag=true
        ;;
    v) 
        voice_change_flag=true
        ;;
    c) 
        config="$OPTARG"
        ;;
    m) 
        model="$OPTARG"
        ;;
    g) 
        gpu="$OPTARG"
        ;;
    h | \?) 
        usage && exit 1
        ;;
    x)
        escape_flag=true
    esac
done


# モード解析
if $training_flag && $voice_change_flag; then
    warn "-t（トレーニングモード） と -v（ボイチェンモード）は同時に指定できません。"
elif $training_flag; then
    echo "■■■  ト レ ー ニ ン グ モ ー ド   ■■■"
elif $voice_change_flag; then
    echo "■■■  ボ イ チ ェ ン モ ー ド  ■■■"
elif $escape_flag; then
    /bin/bash
else
    warn "-t（トレーニングモード） と -v（ボイチェンモード）のいずれかを指定してください。"
fi



if $training_flag; then
    if $resume_flag; then
        echo "トレーニングを再開します"
        docker run -it --gpus all --shm-size=128M \
            -v `pwd`/exp/${name}/dataset:/MMVC_Trainer/dataset \
            -v `pwd`/exp/${name}/logs:/MMVC_Trainer/logs \
            -v `pwd`/exp/${name}/filelists:/MMVC_Trainer/filelists \
            -v `pwd`/vc_resources:/resources \
            -e LOCAL_UID=$(id -u $USER) \
            -e LOCAL_GID=$(id -g $USER) \
            -p ${TENSORBOARD_PORT}:6006 $DOCKER_IMAGE -t -b ${batch_size} -r
    else
        echo "トレーニングを開始します"
        docker run -it --gpus all --shm-size=128M \
            -v `pwd`/exp/${name}/dataset:/MMVC_Trainer/dataset \
            -v `pwd`/exp/${name}/logs:/MMVC_Trainer/logs \
            -v `pwd`/exp/${name}/filelists:/MMVC_Trainer/filelists \
            -v `pwd`/vc_resources:/resources \
            -e LOCAL_UID=$(id -u $USER) \
            -e LOCAL_GID=$(id -g $USER) \
            -p ${TENSORBOARD_PORT}:6006 $DOCKER_IMAGE -t -b ${batch_size}
    fi
fi

if $voice_change_flag; then
    if [[ -z "$config" ]]; then
        warn "コンフィグファイル(-c)を指定してください"
    fi
    if [[ -z "$model" ]]; then
        warn "モデルファイル(-m)を指定してください"
    fi
    if [ "${gpu}" = "on" ]; then
        echo "GPUをマウントして起動します。"

        docker run -it --gpus all --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -p ${VOICE_CHANGER_PORT}:8080 $DOCKER_IMAGE -v -c ${config} -m ${model}
    elif [ "${gpu}" = "off" ]; then
        echo "CPUのみで稼働します。GPUは使用できません。"
        docker run -it --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -p ${VOICE_CHANGER_PORT}:8080 $DOCKER_IMAGE -v -c ${config} -m ${model}
    else
        echo ${gpu}
        warn "-g は onかoffで指定して下さい。"
        
    fi


fi


