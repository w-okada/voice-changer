#!/bin/bash

EXP_NAME=$1

echo $EXP_NAME

# (A)
mkdir -p exp/${EXP_NAME}/logs
mkdir -p exp/${EXP_NAME}/filelists

mkdir -p exp/${EXP_NAME}/dataset
echo "00_myvoice|107"          >  exp/${EXP_NAME}/dataset/multi_speaker_correspondence.txt
echo "01_target_zundamon|100"  >> exp/${EXP_NAME}/dataset/multi_speaker_correspondence.txt
echo "02_target_tsumugi|103"  >> exp/${EXP_NAME}/dataset/multi_speaker_correspondence.txt
echo "03_target_metan|102"  >> exp/${EXP_NAME}/dataset/multi_speaker_correspondence.txt
echo "04_target_ksora|101"  >> exp/${EXP_NAME}/dataset/multi_speaker_correspondence.txt

# (B) トレーニングデータ作成
# (B-0) my voice
mkdir -p exp/${EXP_NAME}/dataset/textful/00_myvoice/text
mkdir -p exp/${EXP_NAME}/dataset/textful/00_myvoice/wav
cp dataset/00_myvoice/wav/*   exp/${EXP_NAME}/dataset/textful/00_myvoice/wav/
cp dataset/00_myvoice/text/*  exp/${EXP_NAME}/dataset/textful/00_myvoice/text/


# (B-1) ずんだもん
mkdir -p exp/${EXP_NAME}/dataset/textful/01_target_zundamon/
unzip -j dataset/1225_zundamon.zip 1225_zundamon/wav/* -d exp/${EXP_NAME}/dataset/textful/01_target_zundamon/wav/
unzip -j dataset/1225_zundamon.zip 1225_zundamon/text/* -d exp/${EXP_NAME}/dataset/textful/01_target_zundamon/text/

# (B-2) 春日部つむぎ
mkdir -p exp/${EXP_NAME}/dataset/textful/02_target_tsumugi/
unzip -j dataset/344_tsumugi.zip 344_tsumugi/wav/* -d exp/${EXP_NAME}/dataset/textful/02_target_tsumugi/wav/
unzip -j dataset/344_tsumugi.zip 344_tsumugi/text/* -d exp/${EXP_NAME}/dataset/textful/02_target_tsumugi/text/

# (B-3) 四国めたん
mkdir -p exp/${EXP_NAME}/dataset/textful/03_target_metan/
unzip -j dataset/459_methane.zip 459_methane/wav/* -d exp/${EXP_NAME}/dataset/textful/03_target_metan/wav/
unzip -j dataset/459_methane.zip 459_methane/text/* -d exp/${EXP_NAME}/dataset/textful/03_target_metan/text/

# (B-4) 九州そら
mkdir -p exp/${EXP_NAME}/dataset/textful/04_target_ksora/
unzip -j dataset/912_sora.zip 912_sora/wav/* -d exp/${EXP_NAME}/dataset/textful/04_target_ksora/wav/
unzip -j dataset/912_sora.zip 912_sora/text/* -d exp/${EXP_NAME}/dataset/textful/04_target_ksora/text/

## 004_expまで。
# echo $1
# mkdir -p ${EXP_NAME}/00_myvoice/text
# mkdir -p ${EXP_NAME}/00_myvoice/wav
# mkdir -p ${EXP_NAME}/logs
# mkdir -p ${EXP_NAME}/filelists


