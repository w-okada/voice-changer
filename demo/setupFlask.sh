#!/bin/bash

echo config: $1
echo model: $2
cp -r /resources/* .

if [[ -e ./setting.json ]]; then
  cp ./setting.json ../frontend/dist/assets/setting.json
fi
pip install flask
pip install flask_cors
python3 serverFlask.py 8080 $1 $2


