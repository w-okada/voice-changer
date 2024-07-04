# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_dynamic_libs, collect_submodules
import sys
import os.path
import site
import logging

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

backend = os.environ.get('BACKEND', 'cpu')

with open('edition.txt', 'w') as f:
    if backend == 'cpu':
      f.write('CPU')
    elif backend == 'dml':
      f.write('DirectML')
    elif backend == 'cuda':
      f.write('NVIDIA-CUDA')
    elif backend == 'rocm':
      f.write('AMD-ROCm')
    else:
      f.write('-')

datas = [('../client/demo/dist', './dist'), ('./edition.txt', '.')]

if 'BUILD_NAME' in os.environ:
  with open('version.txt', 'w') as f:
      f.write(os.environ['BUILD_NAME'])
  datas += [('./version.txt', '.')]
datas += collect_data_files('onnxscript', include_py_files=True)

binaries = []
if backend == 'dml':
  binaries += collect_dynamic_libs('torch_directml')
binaries += collect_dynamic_libs('onnxruntime')

hiddenimports = ['app']
hiddenimports += collect_submodules('scipy') # Fix "ModuleNotFoundError: No module named 'scipy._lib.*'"
hiddenimports += collect_submodules('onnxruntime') # Fix "ModuleNotFoundError: No module named 'onnxruntime.transformers.*'"

a = Analysis(
    ['client.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./pyinstaller-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MMVCServerSIO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MMVCServerSIO',
)
