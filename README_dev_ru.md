Вот перевод файла `README_dev_en.md` на русский язык:

## Для разработчиков

[Японский](/README_dev_ja.md) [Английский](/README_dev_en.md)

## Требования

- Linux (Ubuntu, Debian) или WSL2 (другие дистрибуции Linux и Mac не тестировались)
- Anaconda

## Подготовка

1. Создайте виртуальную среду Anaconda:

```
$ conda create -n vcclient-dev python=3.10
$ conda activate vcclient-dev
```

2. Клонируйте репозиторий:

```
$ git clone https://github.com/w-okada/voice-changer.git
```

## Для серверных разработчиков

1. Установите необходимые зависимости:

1-1. Для систем без GPU

```
$ python -m pip install -r server/requirements_cpuonly.txt
```

1-2. Для систем с GPU NVIDIA

Пожалуйста, перепишите часть `cu128` в файле `server/requirements_nvidia.txt` в соответствии с вашей версией CUDA.
Значение по умолчанию `cu128` предназначено для CUDA 12.8.

```
--extra-index-url https://download.pytorch.org/whl/cu128

torch==2.7.0+cu128
torchaudio==2.7.0+cu128
```

Затем выполните следующую команду:

```
$ python -m pip install -r server/requirements_nvidia.txt
```

2. Запустите сервер

Запустите сервер с помощью следующей команды. Вы можете указать свои пути к весам моделей.

```
$ python3 MMVCServerSIO.py -p 18888 --https true \
    --content_vec_500 pretrain/checkpoint_best_legacy_500.pt  \
    --content_vec_500_onnx pretrain/content_vec_500.onnx \
    --content_vec_500_onnx_on true \
    --hubert_base pretrain/hubert_base.pt \
    --hubert_base_jp pretrain/rinna_hubert_base_jp.pt \
    --hubert_soft pretrain/hubert/hubert-soft-0d54a1f4.pt \
    --nsf_hifigan pretrain/nsf_hifigan/model \
    --crepe_onnx_full pretrain/crepe_onnx_full.onnx \
    --crepe_onnx_tiny pretrain/crepe_onnx_tiny.onnx \
    --rmvpe pretrain/rmvpe.pt \
    --model_dir model_dir \
    --samples samples.json
```

Откройте браузер (на данный момент поддерживается только Chrome), и вы увидите графический интерфейс.

2-1. Устранение неполадок

(1) OSError: не найдена библиотека PortAudio

Если вы получите сообщение ниже, необходимо установить дополнительную библиотеку:

```
OSError: PortAudio library not found
```

Вы можете установить библиотеку командой:

```
$ sudo apt-get install libportaudio2
$ sudo apt-get install libasound-dev
```

(2) Не запускается! Чертова программа!

Клиент не запускается автоматически. Пожалуйста, откройте браузер и перейдите по URL, отображаемому в консоли. И будьте осторожны со словами.

(3) Не удалось загрузить библиотеку libcudnn_cnn_infer.so.8

При использовании WSL может возникнуть ошибка `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory`. Это часто связано с тем, что путь к библиотеке не установлен. Установите путь с помощью команды ниже. Вы можете добавить эту команду в ваш скрипт запуска, например, в .bashrc.

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

- ссылки:
  - https://qiita.com/cacaoMath/items/811146342946cdde5b83
  - https://github.com/microsoft/WSL/issues/8587

3. Наслаждайтесь разработкой.

### Приложение

1. Windows + Anaconda (не поддерживается)

Используйте conda для установки PyTorch:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Также выполните эти команды:

```
pip install chardet
pip install numpy==1.24.0
```

## Для клиентских разработчиков

1. Импорт модулей и начальная сборка:

```
cd client
cd lib
npm install
npm run build:dev
cd ../demo
npm install
npm run build:dev
```

2. Наслаждайтесь.
