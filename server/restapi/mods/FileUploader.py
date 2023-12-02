import os
import shutil
from fastapi import UploadFile


def sanitize_filename(filename: str) -> str:
    safe_filename = os.path.basename(filename)

    max_length = 255
    if len(safe_filename) > max_length:
        file_root, file_ext = os.path.splitext(safe_filename)
        safe_filename = file_root[: max_length - len(file_ext)] + file_ext

    return safe_filename


def upload_file(upload_dirname: str, file: UploadFile, filename: str):
    if file and filename:
        fileobj = file.file
        filename = sanitize_filename(filename)
        target_path = os.path.join(upload_dirname, filename)
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        upload_dir = open(target_path, "wb+")
        shutil.copyfileobj(fileobj, upload_dir)
        upload_dir.close()

        return {"status": "OK", "msg": f"uploaded files {filename} "}
    return {"status": "ERROR", "msg": "uploaded file is not found."}


def concat_file_chunks(upload_dirname: str, filename: str, chunkNum: int, dest_dirname: str):
    filename = sanitize_filename(filename)
    target_path = os.path.join(upload_dirname, filename)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(target_path):
        os.remove(target_path)
    with open(target_path, "ab") as out:
        for i in range(chunkNum):
            chunkName = f"{filename}_{i}"
            chunk_file_path = os.path.join(upload_dirname, chunkName)
            stored_chunk_file = open(chunk_file_path, "rb")
            out.write(stored_chunk_file.read())
            stored_chunk_file.close()
            os.remove(chunk_file_path)
        out.close()
    return {"status": "OK", "msg": f"concat files {out} "}
