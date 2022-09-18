#!/bin/bash

CONFIG=$1
MODEL=$2
TYPE=$3

echo config: $CONFIG
echo model: $MODEL
echo type: $TYPE



cp -r /resources/* .

if [[ -e ./setting.json ]]; then
  cp ./setting.json ../frontend/dist/assets/setting.json
fi

if [ "${TYPE}" = "SOFT_VC" ] ; then
  echo "SOFT_VCを起動します"
  python3 SoftVcServerFlask.py 8080
elif [ "${TYPE}" = "SOFT_VC_FAST_API" ] ; then
  echo "SOFT_VC_FAST_APIを起動します"
  python3 SoftVcServerFastAPI.py 8080
else
  echo "MMVCを起動します"
  python3 serverSIO.py 8080 $CONFIG $MODEL
fi



