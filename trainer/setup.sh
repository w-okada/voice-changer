#!/bin/bash

BATCH_SIZE=$1
RESUME=$2
echo batch:${BATCH_SIZE}
echo resume:${RESUME}

python3 create_dataset_jtalk.py -f train_config -s 24000 -m dataset/multi_speaker_correspondence.txt

sed -ie 's/80000/8000/' train_ms.py
sed -ie "s/\"batch_size\": 10/\"batch_size\": $BATCH_SIZE/" configs/train_config.json


# cd monotonic_align/ \
#  && cythonize -3 -i core.pyx \
#  && mv core.cpython-39-x86_64-linux-gnu.so monotonic_align/ \
#  && cd -

python3 -m tensorboard.main --logdir logs --port 6006 --host 0.0.0.0 &
python3 train_ms.py -c configs/train_config.json -m 20220306_24000 -fg fine_model/G_180000.pth -fd fine_model/D_180000.pth