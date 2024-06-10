# VCClient version 2.0.z マニュアル

## version 体系

バージョンは x.y.z-a 形式で記載する。

| #   | バージョン         | 更新タイミング                      |
| --- | ------------------ | ----------------------------------- |
| x   | メジャーバージョン | 大幅な変更                          |
| y   | マイナーバージョン | 機能追加、VCTypeの追加など          |
| z   | パッチバージョン   | バグ修正。軽微な機能追加            |
| a   | alpha, beta        | aplha version, beta version等に付与 |


## edition

| edition             | os  | 内容                                                                                   |
| ------------------- | --- | -------------------------------------------------------------------------------------- |
| win_std             | win | 一般的なwinユーザ向け。AMD, NvidiaのGPU所有者。CPUのみのユーザ                         |
| win_cuda            | win | NvidiaのGPU所有者向け。cuda, cudnnのセットアップが可能なユーザ                         |
| win_std_torch_dml   | win | pytorchのモデルを使用する場合。AMDのGPU所有者向け。                                    |
| win_cuda_torch_cuda | win | pytorchのモデルを使用する場合。NvidiaのGPU所有者向け。cudaのセットアップが可能なユーザ |
| mac                 | mac | AppleSilicon(M1等)ユーザ向け。                                                         |

## サポート Voice Changer Type
| Voice Chanager Type     | サポートエディション   |                                                                       |
| ----------------------- | ---------------------- | --------------------------------------------------------------------- |
| RVC                     | win_std, win_cuda, mac | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI |
| Beatrice v2 API (alpha) | win_std, win_cuda      | https://prj-beatrice.com/                                             |


## ダウンロード
hugging faceのリポジトリからダウンロードしてください。
ファイル名は次のフォーマットになっています。

```
vcclient_<edition>_<version>.zip
```

## インストール
ダウンロードしたファイルを解凍してください。

解凍先のフォルダパスにASCII以外の文字が含まれている場合に起動しない場合があります。


## アンインストール
解凍したフォルダを削除してください。

## 操作
### 起動方法
| 起動ファイル        | サポートエディション | 説明                                           |
| ------------------- | -------------------- | ---------------------------------------------- |
| start_http.bat      | win_*                | 一般的な起動方法                               |
| start_https.bat     | win_*                | リモートからブラウザアクセスする場合の起動方法 |
| start_http.command  | mac_*                | 一般的な起動方法                               |
| start_https.command | mac_*                | リモートからブラウザアクセスする場合の起動方法 |

初回起動時のみ、必要なデータのダウンロードが行われます。しばらくお待ちください。

### 終了方法
ターミナル上で"q"をタイプすることで終了することもできます。
ネイティブクライアントが立ち上がっている場合はネイティブクライアントをクローズすると終了できます。



## 