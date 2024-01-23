# -*- mode: python ; coding: utf-8 -*-
# from PyInstaller.utils.hooks import collect_all
import sys
import os.path
import site
import logging

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

backend = 'cpu' if 'BACKEND' not in os.environ else os.environ['BACKEND']
python_folder = next(folder for folder in site.getsitepackages() if 'site-packages' in folder).replace('\\', '/')
logging.info(python_folder)

datas = [('../client/demo/dist', './dist'), ('./model_dir_static', './model_dir_static')]

if sys.platform == 'win32':
    binaries = [(python_folder + '/onnxruntime/capi/*.dll', 'onnxruntime/capi')]
else:
    binaries = [(python_folder + '/onnxruntime/capi/*.so', 'onnxruntime/capi')]

hiddenimports = ['MMVCServerSIO']
# tmp_ret = collect_all('fairseq')
# datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['MMVCServerSIO.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
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
