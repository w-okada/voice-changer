[Японский](/README_ja.md) [Корейский](/README_ko.md) [Английский](/README_en.md)

## Что нового!
- Мы выпустили продукт-сестру - клиент Text To Speech.
  - Вы можете насладиться генерацией голоса через простой интерфейс.
  - Подробнее [здесь](https://github.com/w-okada/ttsclient).
- Код тренировки Beatrice V2 теперь доступен!
  - [Репозиторий кода тренировки](https://huggingface.co/fierce-cats/beatrice-trainer)
  - [Версия для Colab](https://github.com/w-okada/beatrice-trainer-colab)
- v.2.0.65-beta
  - [HERE](https://github.com/w-okada/voice-changer/tree/v.2)
  - new feature: We have supported Beatrice v2 beta.1, enabling even higher quality voice conversion.

- v.2.0.61-alpha
  - [ЗДЕСЬ](https://github.com/w-okada/voice-changer/tree/v.2)
  - Новые функции:
    - Теперь можно задавать продолжительность перекрёстного затухания.
  - Исправления:
    - Исправлена проблема, при которой неиспользуемые элементы модели влияли на производительность при объединении моделей, путём установки их значений в ноль.
- v.2.0.60-alpha
  - Новые функции:
    - [Тёмный режим](https://github.com/w-okada/voice-changer/issues/1306)
    - [Возвращение PyTorch RMVPE](https://github.com/w-okada/voice-changer/issues/1319)
    - [Выбор эксклюзивного режима WASAPI](https://github.com/w-okada/voice-changer/issues/1305)
- v.2.0.58-alpha
  - [ЗДЕСЬ](https://github.com/w-okada/voice-changer/tree/v.2)
  - Новые функции:
    - SIO Broadcasting
    - Встроенный ngrok (экспериментальный)
  - Улучшения:
    - Оптимизация для мобильных телефонов.
  - Исправления:
    - Неправильное отображение сообщений CUI на macOS
- v.2.0.55-alpha
  - [ЗДЕСЬ](https://github.com/w-okada/voice-changer/tree/v.2)
  - Улучшения:
    - Снижение нагрузки на процессор для RVC
    - Поддержка WebSocket
  - Изменения:
    - Включение опции `no_cui` в стартовом батч-файле

# Что такое VC Клиент

1. Это клиентское ПО для выполнения преобразования голоса в реальном времени с использованием различных AI для преобразования голоса. Поддерживаемые AI:
   - [MMVC](https://github.com/isletennos/MMVC_Trainer) (только v1)
   - [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) (только v1)
   - [RVC (Retrieval-based Voice Conversion)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
   - [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) (только v1)
   - [Beatrice JVS Corpus Edition](https://prj-beatrice.com/) * экспериментальный * (не по лицензии MIT, см. [readme](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/)), только для Windows, зависит от процессора (только v1)
   - [Beatrice v2](https://prj-beatrice.com/) (только v2)

2. Распределение нагрузки между разными ПК
   Реализация преобразования голоса работает по схеме "сервер-клиент". Вы можете запустить сервер MMVC на отдельном ПК для минимизации влияния на другие ресурсоёмкие процессы, такие как стриминг.

![image](https://user-images.githubusercontent.com/48346627/206640768-53f6052d-0a96-403b-a06c-6714a0b7471d.png)

3. Кроссплатформенная совместимость
   Поддержка Windows, Mac (включая Apple Silicon M1), Linux и Google Colaboratory.

# Как использовать

Это приложение для изменения голоса с использованием MMVC и so-vits-svc.

Есть два основных способа использования, в порядке сложности:

- Использование готового исполняемого файла
- Настройка окружения с Docker или Anaconda

## (1) Использование готовых исполняемых файлов

- Вы можете скачать и запустить исполняемые файлы.

- Смотрите [здесь](tutorials/tutorial_rvc_en_latest.md) для получения руководства. ([устранение неполадок](https://github.com/w-okada/voice-changer/blob/master/tutorials/trouble_shoot_communication_ja.md))

- Теперь попробовать можно на [Google Colaboratory](https://github.com/w-okada/voice-changer/tree/v.2/w_okada's_Voice_Changer_version_2_x.ipynb) (требуется аккаунт ngrok). Вы можете запустить его через кнопку "Открыть в Colab" в верхнем левом углу.

<img src="https://github.com/w-okada/voice-changer/assets/48346627/3f092e2d-6834-42f6-bbfd-7d389111604e" width="400" height="150">

- Мы предлагаем версии для Windows и Mac на [hugging face](https://huggingface.co/wok000/vcclient000/tree/main)
- v2 для Windows
  - Пожалуйста, скачайте и используйте `vcclient_win_std_xxx.zip`. Преобразование голоса можно выполнять с использованием мощного процессора без GPU или с использованием DirectML для GPU (AMD, Nvidia). v2 поддерживает как torch, так и onnx.
  - Если у вас Nvidia GPU, скачайте `vcclient_win_cuda_xxx.zip` для более быстрого преобразования.
- v2 для Mac (Apple Silicon)
  - Пожалуйста, скачайте и используйте `vcclient_mac_xxx.zip`.
- v1
  - Для Windows с Nvidia GPU скачайте ONNX (cpu, cuda), PyTorch (cpu, cuda).
  - Для Windows с AMD/Intel GPU скачайте ONNX (cpu, DirectML) и PyTorch (cpu, cuda). AMD/Intel GPU поддерживаются только для ONNX моделей.

- Для пользователей Windows: после распаковки zip-файла запустите соответствующий `start_http.bat` файл.

- Для Mac: после распаковки zip-файла дважды щёлкните на `startHttp.command`. Если появится сообщение о невозможности проверки разработчика, нажмите Ctrl и повторно запустите.

- Если подключаетесь удалённо, используйте `.command` (Mac) или `.bat` (Windows) файл с https вместо http.

- Энкодер DDPS-SVC поддерживает только hubert-soft.

- [Скачать с hugging face](https://huggingface.co/wok000/vcclient000/tree/main)

## (2) Использование после настройки окружения с Docker или Anaconda

Клонируйте этот репозиторий и используйте его. Для Windows требуется настройка WSL2. Для Mac нужно настроить виртуальные среды Python, например Anaconda. Этот метод обеспечивает наивысшую скорость в большинстве случаев. **<font color="red"> Даже без GPU можно получить достаточную производительность на современном процессоре </font>(смотрите раздел о производительности в реальном времени ниже)**.

[Видео-инструкция по установке WSL2 и Docker](https://youtu.be/POo_Cg0eFMU)

[Видео-инструкция по установке WSL2 и Anaconda](https://youtu.be/fba9Zhsukqw)

Для запуска Docker смотрите [start docker](docker_vcclient/README_en.md).

Для запуска на Anaconda venv смотрите [руководство разработчика](README_dev_ru.md).

Для запуска на Linux с AMD GPU смотрите [руководство](tutorials/tutorial_anaconda_amd_rocm.md).

# Подпись программного обеспечения

Это ПО не подписано разработчиком. Появится предупреждение, но его можно запустить, нажав на иконку с удержанием клавиши Ctrl. Это связано с политикой безопасности Apple. Использование ПО на ваш риск.

![image](https://user-images.githubusercontent.com/48346627/212567711-c4a8d599-e24c-4fa3-8145-a5df7211f023.png)

https://user-images.githubusercontent.com/48346627/212569645-e30b7f4e-079d-4504-8cf8-7816c5f40b00.mp4

# Благодарности

- [Материалы Tachizunda-mon](https://seiga.nicovideo.jp/seiga/im10792934)
- [Irasutoya](https://www.irasutoya.com/)
- [Tsukuyomi-chan](https://tyc.rei-yumesaki.net)

> Это ПО использует голосовые данные бесплатного материала персонажа "Цукуёми-тян", предоставленного CV. Юмесаки Рэй.
>
> - Корпус Цукуёми-тян (CV. Юмесаки Рэй)
>
> https://tyc.rei-yumesaki.net/material/corpus/
>
> Авторское право. Юмесаки Рэй, Все права защищены.

