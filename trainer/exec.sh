#!/bin/bash

# 参考:https://programwiz.org/2022/03/22/how-to-write-shell-script-for-option-parsing/

#set -eux
set -eu

MODE=$1
PARAMS=${@:2:($#-1)}


echo "------"
echo "$MODE"
echo "PARAMS: $PARAMS"
echo "VERBOSE: $VERBOSE"
echo "------"


# 起動
if  [ "${MODE}" = "MMVC" ] ; then
    cd /voice-changer-internal/voice-change-service

    if [[ -e /resources/setting.json ]]; then
        echo "指定された設定(setting.json)を使用します。"
        cp /resources/setting.json ../frontend/dist/assets/setting.json
    else
        echo "デフォルトの設定(setting.json)を使用します。"
        cp ../frontend/dist/assets/setting_mmvc.json ../frontend/dist/assets/setting.json
    fi

    find /resources/ -type f -name "config.json" | xargs -I{} sh -c 'echo "config.jsonをコピーします。" && cp {} ./'
    find /resources/ -type f -name "*.pth"       | xargs -I{} sh -c 'echo "`basename {}`をコピーします。" && cp {} ./'

    if [ "${VERBOSE}" = "on" ]; then
        echo "MMVCを起動します(verbose)"
        python3 MMVCServerSIO.py $PARAMS
    else
        echo "MMVCを起動します"
        python3 MMVCServerSIO.py $PARAMS 2>stderr.txt
    fi
    
elif [ "${MODE}" = "MMVC_TRAIN" ] ; then
    python3 create_dataset_jtalk.py -f train_config -s 24000 -m dataset/multi_speaker_correspondence.txt
    # date_tag=`date +%Y%m%d%H%M%S`
    sed -ie 's/80000/8000/' train_ms.py
    sed -ie "s/\"batch_size\": 10/\"batch_size\": $batch_size/" configs/train_config.json
    sed -ie "s/torch.cuda.device_count()/1/" train_ms.py
    python3 -m tensorboard.main --logdir logs --port 6006 --host 0.0.0.0 &
    python3 train_ms.py $PARAMS
    # if ${resume_flag}; then
    #     echo "トレーニング再開。バッチサイズ: ${batch_size}。"
    #     python3 train_ms.py -c configs/train_config.json -m vc
    # else
    #     echo "トレーニング開始。バッチサイズ: ${batch_size}。"
    #     python3 train_ms.py -c configs/train_config.json -m vc -fg fine_model/G_180000.pth -fd fine_model/D_180000.pth
    # fi
fi


