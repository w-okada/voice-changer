#!/bin/bash

set -eu

if [ $# = 0 ]; then
    echo "
    usage:
        $0 -t <TYPE> <params...>
        TYPE: select one of ['TRAIN', 'MMVC']
    " >&2
    exit 1
fi


# TYPE=$1
# PARAMS=${@:2:($#-1)}

# echo $TYPE
# echo $PARAMS


if [ -e /resources ]; then
  echo "/resources の中身をコピーします。"
  cp -r /resources/* .
else
  echo "/resourcesが存在しません。デフォルトの動作をします。"
fi


## Config 設置
if [[ -e ./setting.json ]]; then
  echo "カスタムセッティングを使用"
  cp ./setting.json ../frontend/dist/assets/setting.json
fi

echo "起動します" "$@"
python3 MMVCServerSIO.py "$@"

### 
# 起動パラメータ
# (1) トレーニングの場合
# python3 MMVCServerSIO.py <type> 
# 環境変数: 
# ※ Colabの場合：python3 MMVCServerSIO.py -t Train -p {PORT} --colab True
# (2) VCの場合


# # 起動
# if [ "${TYPE}" = "MMVC" ] ; then

# elif [ "${TYPE}" = "MMVC_VERBOSE" ] ; then
#   echo "MMVCを起動します(verbose)"
#   python3 MMVCServerSIO.py $PARAMS
# fi


