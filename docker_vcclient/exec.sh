#!/bin/bash

set -eu

# cp -r /weights/* /voice-changer/server/
# /bin/bash

python3 MMVCServerSIO.py $@
# python3 MMVCServerSIO.py -p 18888 --https true \
#   --content_vec_500 checkpoint_best_legacy_500.pt \
#   --hubert_base hubert_base.pt \
#   --hubert_soft hubert-soft-0d54a1f4.pt \
#   --nsf_hifigan nsf_hifigan/model


# -p 18888 --https true \
#     --content_vec_500 checkpoint_best_legacy_500.pt \
#     --hubert_base hubert_base.pt \
#     --hubert_soft hubert-soft-0d54a1f4.pt \
#     --nsf_hifigan nsf_hifigan/model