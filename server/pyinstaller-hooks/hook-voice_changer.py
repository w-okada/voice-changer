from PyInstaller.utils.hooks import collect_all
module_collection_mode = 'py+pyz'
datas, binaries, hiddenimports = collect_all('voice_changer')
