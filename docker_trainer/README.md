MMVC Server
----
# 起動方法

(1) Datasetを`trainer/dataset`におく
```sh
trainer/dataset/
├── 00_myvoice
│   ├── text
│   │   ├── emotion001.txt
│   │   ├── emotion002.txt
...
│   │   └── emotion100.txt
│   └── wav
│       ├── emotion001.wav
│       ├── emotion002.wav
...
│       └── emotion100.wav
├── 1205_zundamon
│   ├── text
│   │   ├── emoNormal_001.txt
│   │   ├── emoNormal_002.txt
...
│   │   └── emoNormal_100.txt
│   └── wav
│       ├── emoNormal_001.wav
│       ├── emoNormal_002.wav
...
│       └── emoNormal_100.wav
├── 344_tsumugi
│   ├── text
│   │   ├── VOICEACTRESS100_001.txt
│   │   ├── VOICEACTRESS100_002.txt
...
│   │   └── emoNormal_100.txt
│   └── wav
│       ├── VOICEACTRESS100_001.wav
│       ├── VOICEACTRESS100_002.wav
...
│       └── emoNormal_100.wav
└── multi_speaker_correspondence.txt
```

(2) start_trainer.shをrootにコピー

(3) `bash start_trainer.sh`を実行

(4) Docker内で次のコマンドを実行
batch sizeは適宜調整
```sh
$ cp configs_org/baseconfig.json configs/
$ python3 normalize.py True
$ python3 create_dataset.py -f train_config -s 24000 -m dataset/multi_speaker_correspondence.txt
$ tensorboard --logdir logs --port 5000 --bind_all &
# batch size 変更
$ python3 train_ms.py -c configs/train_config.json -m 20220306_24000 -fg fine_model/G_v15_best.pth -fd fine_model/D_v15_best.pth


$ python3 train_ms.py -c configs/train_config.json -m 20220306_24000
```

(x) テスト
```
$ python3 MMVC_Client/python/conver_test.py -m logs/G_40000.pth -c configs/train_config.json -s 0 -t 101 --input dataset/00_myvoice/wav/emotion011.wav --output dataset/test.wav --f0_scale 3
```


(X) onnx
python3 onnx_export.py  --config_file logs/train_config.json  --convert_pth logs/G_220000.pth