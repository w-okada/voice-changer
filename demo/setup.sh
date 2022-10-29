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
  cp ../frontend/dist/assets/setting_mmvc.json ../frontend/dist/assets/setting.json  
fi


# 起動
if [ "${TYPE}" = "MMVC" ] ; then
  echo "MMVCを起動します"
  python3 MMVCServerSIO.py $PARAMS 2>stderr.txt
elif [ "${TYPE}" = "MMVC_VERBOSE" ] ; then
  echo "MMVCを起動します(verbose)"
  python3 MMVCServerSIO.py $PARAMS
fi


