#!/bin/bash
set -eu

DOCKER_IMAGE=dannadori/voice-changer:20221003_002318
#DOCKER_IMAGE=voice-changer


MODE=$1
PARAMS=${@:2:($#-1)}

### DEFAULT VAR ###
DEFAULT_EX_PORT=18888
DEFAULT_USE_GPU=on # on|off
DEFAULT_VERBOSE=off # on|off

### ENV VAR ###
EX_PORT=${EX_PORT:-${DEFAULT_EX_PORT}}
USE_GPU=${USE_GPU:-${DEFAULT_USE_GPU}}
VERBOSE=${VERBOSE:-${DEFAULT_VERBOSE}}

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
        -e VERBOSE=${VERBOSE} \
        -p ${EX_PORT}:8080 $DOCKER_IMAGE "$@"
    else
        echo "MMVCを起動します(only cpu)"
        docker run -it --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -e VERBOSE=${VERBOSE} \
        -p ${EX_PORT}:8080 $DOCKER_IMAGE "$@"

        # docker run -it --shm-size=128M \
        # -v `pwd`/vc_resources:/resources \
        # -e LOCAL_UID=$(id -u $USER) \
        # -e LOCAL_GID=$(id -g $USER) \
        # -e EX_IP="`hostname -I`" \
        # -e EX_PORT=${EX_PORT} \
        # -e VERBOSE=${VERBOSE} \
        # --entrypoint="" \
        # -p ${EX_PORT}:8080 $DOCKER_IMAGE /bin/bash

    fi

elif [ "${MODE}" = "SOFT_VC" ]; then
    if [ "${USE_GPU}" = "on" ]; then
        echo "Start Soft-vc"

        docker run -it --gpus all --shm-size=128M \
        -v `pwd`/vc_resources:/resources \
        -e LOCAL_UID=$(id -u $USER) \
        -e LOCAL_GID=$(id -g $USER) \
        -e EX_IP="`hostname -I`" \
        -e EX_PORT=${EX_PORT} \
        -e VERBOSE=${VERBOSE} \
        -p ${EX_PORT}:8080 $DOCKER_IMAGE "$@"
    else
        echo "Start Soft-vc withou GPU is not supported"
    fi

else
    echo "
usage: 
    $0 <MODE> <params...>
    MODE: select one of ['MMVC_TRAIN', 'MMVC', 'SOFT_VC']
" >&2
fi




# echo $EX_PORT


# echo "------"
# echo "$@"
# echo "------"

# # usage() {
# #     echo "
# # usage: 
# #     For training
# #         $0 [-t] -n <exp_name> [-b batch_size] [-r] 
# #             -t: トレーニングモードで実行する場合に指定してください。(train)
# #             -n: トレーニングの名前です。(name)
# #             -b: バッチサイズです。(batchsize)
# #             -r: トレーニング再開の場合に指定してください。(resume)
# #     For changing voice
# #         $0 [-v] [-c config] [-m model] [-g on/off]
# #             -v: ボイスチェンジャーモードで実行する場合に指定してください。(voice changer)
# #             -c: トレーニングで使用したConfigのファイル名です。(config)
# #             -m: トレーニング済みのモデルのファイル名です。(model)
# #             -g: GPU使用/不使用。デフォルトはonなのでGPUを使う場合は指定不要。(gpu)
# #             -p: port番号
# #     For help
# #         $0 [-h]
# #             -h: show this help
# # " >&2
# # }
# # warn () {
# #     echo "! ! ! $1 ! ! !"
# #     exit 1
# # }


# # training_flag=false
# # name=999_exp
# # batch_size=10
# # resume_flag=false

# # voice_change_flag=false
# # config=
# # model=
# # gpu=on
# # port=8080
# # escape_flag=false

# # # オプション解析
# # while getopts tn:b:rvc:m:g:p:hx OPT; do
# #     case $OPT in
# #     t) 
# #         training_flag=true
# #         ;;
# #     n) 
# #         name="$OPTARG"
# #         ;;
# #     b) 
# #         batch_size="$OPTARG"
# #         ;;
# #     r) 
# #         resume_flag=true
# #         ;;
# #     v) 
# #         voice_change_flag=true
# #         ;;
# #     c) 
# #         config="$OPTARG"
# #         ;;
# #     m) 
# #         model="$OPTARG"
# #         ;;
# #     g) 
# #         gpu="$OPTARG"
# #         ;;
# #     p) 
# #         port="$OPTARG"
# #         ;;
# #     h | \?) 
# #         usage && exit 1
# #         ;;
# #     x)
# #         escape_flag=true
# #     esac
# # done


# # # モード解析
# # if $training_flag && $voice_change_flag; then
# #     warn "-t（トレーニングモード） と -v（ボイチェンモード）は同時に指定できません。"
# # elif $training_flag; then
# #     echo "■■■  ト レ ー ニ ン グ モ ー ド   ■■■"
# # elif $voice_change_flag; then
# #     echo "■■■  ボ イ チ ェ ン モ ー ド  ■■■"
# # elif $escape_flag; then
# #     /bin/bash
# # else
# #     warn "-t（トレーニングモード） と -v（ボイチェンモード）のいずれかを指定してください。"
# # fi

# if [ "${MODE}" = "MMVC_TRAIN_INITIAL" ]; then
#     echo "トレーニングを開始します"
# elif [ "${MODE}" = "MMVC" ]; then
#     echo "MMVCを起動します"

#     docker run -it --gpus all --shm-size=128M \
#     -v `pwd`/vc_resources:/resources \
#     -e LOCAL_UID=$(id -u $USER) \
#     -e LOCAL_GID=$(id -g $USER) \
#     -e EX_IP="`hostname -I`" \
#     -e EX_PORT=${port} \
#     -p ${port}:8080 $DOCKER_IMAGE -v -c ${config} -m ${model}

# elif [ "${MODE}" = "MMVC_VERBOSE" ]; then
#     echo "MMVCを起動します(verbose)"
# elif [ "${MODE}" = "MMVC_CPU" ]; then
#     echo "MMVCを起動します(CPU)"
# elif [ "${MODE}" = "MMVC_CPU_VERBOSE" ]; then
#     echo "MMVCを起動します(CPU)(verbose)"
# elif [ "${MODE}" = "SOFT_VC" ]; then
#     echo "Start Soft-vc"
# elif [ "${MODE}" = "SOFT_VC_VERBOSE" ]; then
#     echo "Start Soft-vc(verbose)"
# else
#     echo "
# usage: 
#     $0 <MODE> <params...>
#     EX_PORT: 
#     MODE: one of ['MMVC_TRAIN', 'MMVC', 'SOFT_VC']

#     For 'MMVC_TRAIN':
#         $0 MMVC_TRAIN_INITIAL -n <exp_name> [-b batch_size] [-r] 
#             -n: トレーニングの名前です。(name)
#             -b: バッチサイズです。(batchsize)
#             -r: トレーニング再開の場合に指定してください。(resume)
#     For 'MMVC'
#         $0 MMVC [-c config] [-m model] [-g on/off] [-p port] [-v]
#             -c: トレーニングで使用したConfigのファイル名です。(config)
#             -m: トレーニング済みのモデルのファイル名です。(model)
#             -g: GPU使用/不使用。デフォルトはonなのでGPUを使う場合は指定不要。(gpu)
#             -p: Docker からExposeするport番号
#             -v: verbose
#     For 'SOFT_VC'
#         $0 SOFT_VC [-c config] [-m model] [-g on/off]
#             -p: port exposed from docker container.
#             -v: verbose
# " >&2
# fi



# # if $training_flag; then
# #     if $resume_flag; then
# #         echo "トレーニングを再開します"
# #         docker run -it --gpus all --shm-size=128M \
# #             -v `pwd`/exp/${name}/dataset:/MMVC_Trainer/dataset \
# #             -v `pwd`/exp/${name}/logs:/MMVC_Trainer/logs \
# #             -v `pwd`/exp/${name}/filelists:/MMVC_Trainer/filelists \
# #             -v `pwd`/vc_resources:/resources \
# #             -e LOCAL_UID=$(id -u $USER) \
# #             -e LOCAL_GID=$(id -g $USER) \
# #             -p ${TENSORBOARD_PORT}:6006 $DOCKER_IMAGE -t -b ${batch_size} -r
# #     else
# #         echo "トレーニングを開始します"
# #         docker run -it --gpus all --shm-size=128M \
# #             -v `pwd`/exp/${name}/dataset:/MMVC_Trainer/dataset \
# #             -v `pwd`/exp/${name}/logs:/MMVC_Trainer/logs \
# #             -v `pwd`/exp/${name}/filelists:/MMVC_Trainer/filelists \
# #             -v `pwd`/vc_resources:/resources \
# #             -e LOCAL_UID=$(id -u $USER) \
# #             -e LOCAL_GID=$(id -g $USER) \
# #             -p ${TENSORBOARD_PORT}:6006 $DOCKER_IMAGE -t -b ${batch_size}
# #     fi
# # fi

# # if $voice_change_flag; then
# #     if [[ -z "$config" ]]; then
# #         warn "コンフィグファイル(-c)を指定してください"
# #     fi
# #     if [[ -z "$model" ]]; then
# #         warn "モデルファイル(-m)を指定してください"
# #     fi
# #     if [ "${gpu}" = "on" ]; then
# #         echo "GPUをマウントして起動します。"

# #         docker run -it --gpus all --shm-size=128M \
# #         -v `pwd`/vc_resources:/resources \
# #         -e LOCAL_UID=$(id -u $USER) \
# #         -e LOCAL_GID=$(id -g $USER) \
# #         -e EX_IP="`hostname -I`" \
# #         -e EX_PORT=${port} \
# #         -p ${port}:8080 $DOCKER_IMAGE -v -c ${config} -m ${model}
# #     elif [ "${gpu}" = "off" ]; then
# #         echo "CPUのみで稼働します。GPUは使用できません。"
# #         docker run -it --shm-size=128M \
# #         -v `pwd`/vc_resources:/resources \
# #         -e LOCAL_UID=$(id -u $USER) \
# #         -e LOCAL_GID=$(id -g $USER) \
# #         -e EX_IP="`hostname -I`" \
# #         -e EX_PORT=${port} \
# #         -p ${port}:8080 $DOCKER_IMAGE -v -c ${config} -m ${model}
# #     else
# #         echo ${gpu}
# #         warn "-g は onかoffで指定して下さい。"
        
# #     fi


# # fi


