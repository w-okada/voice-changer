「4 MMVCの学習に必要なconfig系Fileを作成する」でエラーが発生する
===========================================================================

テキストファイルがUTF-8形式でない場合
---------------------------------------------------------------------------
下記は学習データのテキストファイルがUTF-8形式でない場合に生じるエラーログになります。 ::

   Traceback (most recent call last):
   File "create_dataset_jtalk.py", line 103, in <module>
      main()
   File "create_dataset_jtalk.py", line 99, in main
      n_spk = create_dataset(filename)
   File "create_dataset_jtalk.py", line 49, in create_dataset
      mozi = f.read().split("\n")
   File "/usr/lib/python3.7/codecs.py", line 322, in decode
      (result, consumed) = self._buffer_decode(data, self.errors, final)
   UnicodeDecodeError: 'utf-8' codec can't decode byte 0x83 in position 0: invalid start byte

| このエラーはテキストファイルがUTF-8形式でない場合に出ると思われます。
| 以下の方法で学習に使用するテキストファイルの形式をご確認ください。
| 1. テキストファイルをメモ帳で開く
| 2. ファイル(F) -> 名前を付けて保存(A)を選択
| 3. ファイル名、ファイルの種類の下にある文字コード(E)が「UTF-8」になっているか確認する


「6 学習を実行します」でエラーが発生する
=============================================
ファインチューニング用のモデルを読み込みの失敗場合
---------------------------------------------------------------------------
下記はファインチューニング用のモデルを読み込みに失敗したときのエラーログになります。 ::

   File "train_ms.py", line 303, in <module>
      main()
   File "train_ms.py", line 53, in main
      mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 200, in spawn
      return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 158, in start_processes
      while not context.join():
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 119, in join
      raise Exception(msg)
   Exception: 

   -- Process 0 terminated with the following error:
   Traceback (most recent call last):
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
      fn(i, *args)
   File "/content/drive/MyDrive/MMVC_Trainer-main/train_ms.py", line 108, in run
      _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_g, net_g, optim_g)
   File "/content/drive/MyDrive/MMVC_Trainer-main/utils.py", line 19, in load_checkpoint
      assert os.path.isfile(checkpoint_path)
   AssertionError
   
| ファインチューニング用のモデルの読み込みに失敗しています。
| /finemodel 配下に finemodel用のモデルファイル(.pth)がありますか？
| (v1.1.0 では G_232000.pthとD_232000.pth)


detasetにデータが正しく配置されていない場合
---------------------------------------------------------------------------
下記はdetasetにデータが正しく配置されていないときのエラーログになります。

size mismatch for emb_g.weight:とエラーが出た場合、ほぼ確実にdetasetの配置ミスです。 ::

   -- Process 0 terminated with the following error:
   Traceback (most recent call last):
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
      fn(i, *args)
   File "/content/drive/MyDrive/MMVC_Trainer-main/train_ms.py", line 108, in run
      _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_g, net_g, optim_g)
   File "/content/drive/MyDrive/MMVC_Trainer-main/utils.py", line 38, in load_checkpoint
      model.module.load_state_dict(new_state_dict)
   File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1045, in load_state_dict
      self.__class__.__name__, "\n\t".join(error_msgs)))
   RuntimeError: Error(s) in loading state_dict for SynthesizerTrn:
      size mismatch for emb_g.weight: copying a param with shape torch.Size([106, 256]) from checkpoint, the shape in current model is torch.Size([104, 256]).

| datasetに正しくデータが配置されていません。
| (後で加筆します…)


データのビットレートがあっていない場合
---------------------------------------------------------------------------
下記は学習データのビットレートがあっていない場合に生じるエラーログになります。 ::

      Exception ignored in: <function _MultiProcessingDataLoaderIter.del at 0x7f799c9945f0>
   Traceback (most recent call last):
   File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 1101, in del
      self._shutdown_workers()
   File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 1050, in _shutdown_workers
      if not self._shutdown:
   AttributeError: '_MultiProcessingDataLoaderIter' object has no attribute '_shutdown'
   Traceback (most recent call last):
   File "train_ms.py", line 303, in <module>
      main()
   File "train_ms.py", line 53, in main
      mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 200, in spawn
      return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 158, in start_processes
      while not context.join():
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 113, in join
      (error_index, exitcode)
      Exception: process 0 terminated with exit code 1

| 現在の学習元データは384Kbpx(24000Hz 16bit 1ch)で行われています。
| そのため、自分の声の音声データと変換したい声の音声データがこの条件に合っていない場合、このエラーが発生します。
| 録音ツールは24000Hzで録音できるものが少なく、近い周波数の22050Hzで録音した場合もエラーになるので注意してください。またchを1ch(モノラル)にするよう注意してください。
| Audacityという録音ツールでは録音周波数を任意に指定できるため直接24000Hzで録音することができるのでおすすめです。
| 変換したい声の音声データに「ITAコーパス」や「つくよみちゃんコーパス」を利用する場合もこの条件に合わないため変換する必要があります。
| 変換には yatoyun 氏作成の https://colab.research.google.com/drive/1CXc7B_mZE3mZLm3MPXMNm_NB56mpBPZS などを利用してみてください。


GPUが使えないと言われて学習できない
---------------------------------------------------------------------------
下記のようなエラーが表示されて学習できません。 ::

   AssertionError: CPU training is not allowed.

| Train_MMVC.ipynbの下にある
| ランタイム→ランタイムのタイプの変更→ハードウェアアクセラレータ
| のところを「GPU」に変更して再度実行します。


TPUを使って学習できますか
---------------------------------------------------------------------------
現在はTPUに対応していません。


問題点を修正したのに同じエラーが出る
---------------------------------------------------------------------------
データ修正後 config ファイルを作り直ししないとダメな場合があるので、作り直して再度試してください。


MMVC_Clientでエラーが発生する
==============================================================

ファイルの指定方法が間違ってる場合(ローカル版)
---------------------------------------------------------------------------
以下がrt-mmvc-client-GPU.exeでmyprofile.jsonのパスを入力したときに生じるエラーログです ::

   Traceback (most recent call last):
      File "{あなたのパス}/rt-mmvc-client-GPU.py",line 424,in <module>
      File "{あなたのパス*/rt-mmvc-client-GPU.py",line 402, in config_get
   OSError:[Error 22] Invalid argument: "{あなたのパス}/myprofile.json"

| パスを指定する際はなにもつけずに指定してください。
|


jsonファイルの記法が間違ってる場合(ローカル版)
---------------------------------------------------------------------------
以下はrt-mmvc-client-GPU.exeで、myprofile.jsonに記述されているパス指定が間違っているときに生じるエラーログです。 ::

   Traceback (most recent call last):
      File "{あなたのパス}/rt-mmvc-client-GPU.py",line 424,in <module>
      File "{あなたのパス}/rt-mmvc-client-GPU.py",line 402, in config_get
      File "{あなたのパス}/json/__init__.py",line 346, in loads
      File "{あなたのパス}/json/decoder.py",line 357, in decode
      File "{あなたのパス}/json/decoder.py",line 353, in raw_decode
   json.decoder.JSONDecodeError:Invalid \escape: line 14 column 15 (char 255)

| jsonファイル内ではパスの「\\\」を「\\\\」と表記する必要があります。

path指定の区切り子は下記例のように記述してください。 ::

   "path": {
     "json":"C:\\Users\\example_user\\Programs\\rt-mmvc_client\\config.json",
     "model":"C:\\Users\\example_user\\Programs\\rt-mmvc_client\\G_2000.pth",
     "noise":"C:\\Users\\example_user\\Programs\\rt-mmvc_client\\noise.wav"
   }


CPU版で音声が途切れ途切れになる
---------------------------------------------------------------------------
| 現在のバージョンではかなりCPU負荷がかかるため途切れ途切れになってしまいます。
| GPU版でもCPUの負荷が結構高く逆にGPUの負荷がほとんどかからないためGPU版を利用してください。


データセットについて
=============================================

学習データの自分の声と変換先のテキスト内容は一致させる必要はありますか
---------------------------------------------------------------------------
一致しなくても大丈夫ですが、声優統計コーパスやATR503文(内100文程度で可)などの所謂音素分を読み上げることを推奨します。


どのぐらいの量の自分の声が必要ですか
---------------------------------------------------------------------------
文章量にもよりますが100文程度でも十分な精度がでます。


テキスト内容の記述方法はどうしたらよいですか
---------------------------------------------------------------------------
| 漢字さえ使わなければよいです。
| 基本はひらがなかカナに統一したほうがよいです。句読点はあったほうがよいです。


推奨されるデータ、ボイスチェンジャーを使用する際のマイクの諸設定を教えてください
---------------------------------------------------------------------------------------------------------
| 学習目標の声の音声ファイルと自分の声の音声ファイルはすべて同じサンプリングレート、bit、チャンネル数(すべてをかけ合わせて算出されるビットレート)にする必要があります。
| ボイスチェンジャーを使用する際の設定は学習時に使用した音声のサンプリングレート、bit、チャンネル数と合わせてください。


その他
========

最終更新:2021/04/20