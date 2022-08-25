#!/bin/bash

docker run -it --shm-size=128M \
  -v `pwd`/vc_resources:/resources \
  -e LOCAL_UID=$(id -u $USER) \
  -e LOCAL_GID=$(id -g $USER) \
  -p 6006:6006 -p 8080:8080 dannadori/voice-changer:20220826_023623 "$@"
