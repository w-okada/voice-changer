#!/bin/bash

TYPE=$1
CONFIG=$2
MODEL=$3

echo type: $TYPE
echo config: $CONFIG
echo model: $MODEL



cp -r /resources/* .

## Config 設置
if [[ -e ./setting.json ]]; then
  echo "カスタムセッティングを使用"
  cp ./setting.json ../frontend/dist/assets/setting.json
else
  if [ "${TYPE}" = "SOFT_VC" ] ; then
    cp ../frontend/dist/assets/setting_softvc.json ../frontend/dist/assets/setting.json
  elif [ "${TYPE}" = "SOFT_VC_FAST_API" ] ; then
    cp ../frontend/dist/assets/setting_softvc_colab.json ../frontend/dist/assets/setting.json
  else
    cp ../frontend/dist/assets/setting_mmvc.json ../frontend/dist/assets/setting.json  
  fi
fi

# 起動
if [ "${TYPE}" = "SOFT_VC" ] ; then
  echo "SOFT_VCを起動します"
  python3 SoftVcServerSIO.py 8080
elif [ "${TYPE}" = "SOFT_VC_FAST_API" ] ; then
  echo "SOFT_VC_FAST_APIを起動します"
  python3 SoftVcServerFastAPI.py 8080 docker
else
  echo "MMVCを起動します"
  python3 serverSIO.py 8080 $CONFIG $MODEL
fi



