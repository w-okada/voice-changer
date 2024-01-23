# -*- mode: python ; coding: utf-8 -*-
# from PyInstaller.utils.hooks import collect_all
import sys
import os.path

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

backend = 'cpu' if 'BACKEND' not in os.environ else os.environ['BACKEND']
python_folder = os.path.dirname(sys.executable.replace('\\', '/'))

datas = [('../client/demo/dist', './dist'), ('./model_dir_static', './model_dir_static')]

if backend == 'dml':
    if sys.platform == 'win32':
        binaries = [(python_folder + '/Lib/site-packages/torch_directml/DirectML.dll', 'torch_directml'), (python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll', 'onnxruntime/capi')]
    else:
        raise Exception('DML is not available for other platforms.')
elif backend == 'cpu':
    if sys.platform == 'win32':
        binaries = [(python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll', 'onnxruntime/capi')]
    else:
        binaries = [(python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.so', 'onnxruntime/capi')]
elif backend == 'cuda':
    if sys.platform == 'win32':
        binaries = [(python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll', 'onnxruntime/capi'), (python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll', 'onnxruntime/capi')]
    else:
        binaries = [(python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.so', 'onnxruntime/capi'), (python_folder + '/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.so', 'onnxruntime/capi')]

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
