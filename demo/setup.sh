#!/bin/bash
cp -r /resources/* .

TYPE=$1
PARAMS=${@:2:($#-1)}

echo $TYPE
echo $PARAMS

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
  python3 SoftVcServerSIO.py $PARAMS 2>stderr.txt
elif [ "${TYPE}" = "SOFT_VC_VERBOSE" ] ; then
  echo "SOFT_VCを起動します(verbose)"
  python3 SoftVcServerSIO.py $PARAMS 
elif [ "${TYPE}" = "SOFT_VC_FAST_API" ] ; then
  echo "SOFT_VC_FAST_APIを起動します"
  python3 SoftVcServerFastAPI.py 8080 docker
elif [ "${TYPE}" = "MMVC" ] ; then
  echo "MMVCを起動します"
  python3 serverSIO.py $PARAMS 2>stderr.txt
elif [ "${TYPE}" = "MMVC_VERBOSE" ] ; then
  echo "MMVCを起動します(verbose)"
  python3 serverSIO.py $PARAMS
fi


