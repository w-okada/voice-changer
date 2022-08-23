#!/bin/bash

# 参考:https://programwiz.org/2022/03/22/how-to-write-shell-script-for-option-parsing/

set -eu

echo "------"
echo "$@"
echo "------"

usage() {
    echo "
usage: 
    For training
        $0 [-t] [-b batch_size] [-r] 
            -t: flag for training mode
            -b: batch_size.
            -r: flag for resuming training.
    For changing voice
        $0 [-v] [-c config] [-m model]
            -v: flag for voice change mode
            -c: config
            -m: model name
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
batch_size=10
resume_flag=false

voice_change_flag=false
config=
model=

escape_flag=false

# オプション解析
while getopts tb:rvc:m:hx OPT; do
    case $OPT in
    t) 
        training_flag=true
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
    h | \?) 
        usage && exit 1
        ;;
    x)
        escape_flag=true
    esac
done



# ## コマンドライン引数から、オプション引数分を削除
# # shift $((OPTIND - 1))

# モード解析
if $training_flag && $voice_change_flag; then
    warn "-t（トレーニングモード） と -v（ボイチェンモード）は同時に指定できません。"
    exit 1
elif $training_flag; then
    echo "■■■  ト レ ー ニ ン グ モ ー ド   ■■■"
elif $voice_change_flag; then
    echo "■■■  ボ イ チ ェ ン モ ー ド  ■■■"
elif $escape_flag; then
    /bin/bash
else
    warn "-t（トレーニングモード） と -v（ボイチェンモード）のいずれかを指定してください。"
    exit 1
fi



# if $training_flag; then


#     python3 create_dataset_jtalk.py -f train_config -s 24000 -m dataset/multi_speaker_correspondence.txt
#     # date_tag=`date +%Y%m%d%H%M%S`
#     sed -ie 's/80000/8000/' train_ms.py
#     sed -ie "s/\"batch_size\": 10/\"batch_size\": $batch_size/" configs/train_config.json

#     python3 -m tensorboard.main --logdir logs --port 6006 --host 0.0.0.0 &

#     if ${resume_flag}; then
#         echo "トレーニング再開。バッチサイズ: ${batch_size}。"
#         python3 train_ms.py -c configs/train_config.json -m vc
#     else
#         echo "トレーニング開始。バッチサイズ: ${batch_size}。"
#         python3 train_ms.py -c configs/train_config.json -m vc -fg fine_model/G_180000.pth -fd fine_model/D_180000.pth
#     fi
# fi

# if $voice_change_flag; then
#     if [[ -z "$config" ]]; then
#         warn "コンフィグファイル(-c)を指定してください"
#     fi
#     if [[ -z "$model" ]]; then
#         warn "モデルファイル(-m)を指定してください"
#     fi

#     cd /voice-changer-internal/voice-change-service

#     cp -r /resources/* .
#     if [[ -e ./setting.json ]]; then
#         cp ./setting.json ../frontend/dist/assets/setting.json
#     fi
#     echo "-----------!!"
#     echo $config $model
#     echo $model
#     python3 serverSIO.py 8080 $config $model
# fi
