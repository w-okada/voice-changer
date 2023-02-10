MMVC Server
----
# 起動方法

```
$ conda create -n mmvc-server python=3.10
$ conda activate mmvc-server
$ pip install -r requirements.txt

$ git clone https://github.com/isletennos/MMVC_Client.git
$ cd MMVC_Client && git checkout 3374a1177b73e3f6d600e5dbe93af033c36ee120 && cd -

$ git clone https://github.com/isletennos/MMVC_Trainer.git
$ cd MMVC_Trainer && git checkout c242d3d1cf7f768af70d9735082ca2bdd90c45f3 && cd -

$ python3 MMVCServerSIO.py -p 18888 --https true 
```

