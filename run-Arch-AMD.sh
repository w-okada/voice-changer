#!/bin/bash
# Condaの初期化w
source /opt/miniforge/etc/profile.d/conda.sh
conda activate rvc

# --- [追加] 自動クリーンアップ設定 ---
# このスクリプトが終了（EXIT）したり、中断（SIGINT/Ctrl+C）された時に
# 自動的にポート18888を掃除する関数を定義します。
cleanup() {
    echo "Stopping Voice Changer and cleaning up port 18888..."
    fuser -k 18888/tcp > /dev/null 2>&1
    exit
}
# 終了信号をキャッチして cleanup を実行するように設定
trap cleanup EXIT SIGINT SIGTERM
# ------------------------------------

# GPU設定
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export MIOPEN_FIND_MODE=2

# メモリ対策（念のため）
export PYTHONMALLOC=malloc
export MALLOC_CHECK_=0

#デバイス
export ALSA_PCM_CARD=default
export PA_ALSA_PLUGHW=1
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\$LD_LIBRARY_PATH

cd ~/software/voice-changer/server || exit

python MMVCServerSIO.py --rest_port 18888 --use_gpu
