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

# Dev
```
# for 1.3
cd MMVC_Client
git checkout 04f3fec4fd82dea6657026ec4e1cd80fb29a415c
cd ..

# for 1.5
cd MMVC_Client
git checkout 6dd4f2451fec701d85f611fa831d7e5f4ddce8da
cd ..

# for so-vits-svc
cd so-vits-svc/
git checkout 016db3de81f6a4034b85ffba120554d07829f132
cd ..

```