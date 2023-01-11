MMVC Server
----
# 起動方法

```
$ conda create -n mmvc-server python=3.9  
$ conda activate mmvc-server
$ pip install -r requirements.txt

$ git clone --depth 1 https://github.com/isletennos/MMVC_Trainer.git -b v1.3.2.2
$ cd MMVC_Trainer/monotonic_align/ && python setup.py build_ext --inplace && cd -
$ python3 MMVCServerSIO.py -p 18888 --https true 
```

