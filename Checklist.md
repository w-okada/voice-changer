# Release Check List
## Run
- [ ] Anaconda on Linux
- [ ] Docker on Linux
- [ ] Anaconda on WSL2
- [ ] Docker on WSL2
- [ ] Colab simple
- [ ] Colab normal
- [ ] Windows exe
- [ ] Mac(M1)

## Doc
- [ ] Readme
- [ ] Wiki
- [ ] Zenn


# Memo
## Release Process
一通り開発が終わったと思ったら.

(1) Dockerを生成
```
npm run build:docker
npm run push:docker
```
Tagをメモ。

(2) start2.shを編集
メモしたTagを貼り付け。
```
bash start2.sh
```

(3) exeファイル作成
(3-1) Win
・環境変数にリリースバージョンを設定
・pipenv

(4) Readmeにリンクをはる

(5) Branch 解除。Tag化

(6) Colabチェック
