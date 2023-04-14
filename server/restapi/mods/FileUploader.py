import os
import shutil
from fastapi import UploadFile

# UPLOAD_DIR = "model_upload_dir"


def upload_file(upload_dirname: str, file: UploadFile, filename: str):
    if file and filename:
        fileobj = file.file
        upload_dir = open(os.path.join(upload_dirname, filename), 'wb+')
        shutil.copyfileobj(fileobj, upload_dir)
        upload_dir.close()

        return {"status": "OK", "msg": f"uploaded files {filename} "}
    return {"status": "ERROR", "msg": "uploaded file is not found."}


def concat_file_chunks(slot: int, upload_dirname: str, filename: str, chunkNum: int, dest_dirname: str):
    # target_dir = os.path.join(dest_dirname, f"{slot}")
    target_dir = os.path.join(dest_dirname)
    os.makedirs(target_dir, exist_ok=True)
    target_file_name = os.path.join(target_dir, filename)
    if os.path.exists(target_file_name):
        os.remove(target_file_name)
    with open(target_file_name, "ab") as target_file:
        for i in range(chunkNum):
            chunkName = f"{filename}_{i}"
            chunk_file_path = os.path.join(upload_dirname, chunkName)
            stored_chunk_file = open(chunk_file_path, 'rb')
            target_file.write(stored_chunk_file.read())
            stored_chunk_file.close()
            os.remove(chunk_file_path)
        target_file.close()
    return {"status": "OK", "msg": f"concat files {target_file_name} "}
