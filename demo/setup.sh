#!/bin/bash

echo config: $1
echo model: $2
cp -r /resources/* .

if [[ -e ./setting.json ]]; then
  cp ./setting.json ../frontend/dist/assets/setting.json
fi

python3 serverSIO.py 8080 $1 $2


