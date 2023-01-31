#!/bin/bash

data_tag=`date +%Y%m%d_%H%M%S`
docker login 

docker tag trainer dannadori/trainer:$data_tag
docker push dannadori/trainer:$data_tag
