MMVC Server
----
# 起動方法

```
$ conda create -n mmvc-server python=3.10
$ conda activate mmvc-server
$ pip install -r requirements.txt

$ git clone https://github.com/isletennos/MMVC_Client.git
$ cd MMVC_Client && git checkout 04f3fec4fd82dea6657026ec4e1cd80fb29a415c && cd -
$ python3 MMVCServerSIO.py -p 18888 --https true 
```

