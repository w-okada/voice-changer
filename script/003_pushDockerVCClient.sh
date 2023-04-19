#!/bin/bash

data_tag=`date +%Y%m%d_%H%M%S`
docker login 

docker tag vcclient dannadori/vcclient:$data_tag
docker push dannadori/vcclient:$data_tag
