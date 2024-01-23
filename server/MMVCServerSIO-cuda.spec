# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['MMVCServerSIO.py'],
    pathex=[],
    binaries=[('C:/Users/Snek/AppData/Local/Programs/Python/Python310/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll', 'onnxruntime/capi'), ('C:/Users/Snek/AppData/Local/Programs/Python/Python310/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll', 'onnxruntime/capi')],
    datas=[('../client/demo/dist', './dist'), ('./model_dir_static', './model_dir_static')],
    hiddenimports=['MMVCServerSIO'],
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
