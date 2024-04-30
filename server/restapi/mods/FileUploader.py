import os
import shutil
from fastapi import UploadFile

MAX_PATH_LENGTH = 255

def sanitize_filename(filename: str) -> str:
    safe_filename = os.path.basename(filename)

    if len(safe_filename) > MAX_PATH_LENGTH:
        file_root, file_ext = os.path.splitext(safe_filename)
        safe_filename = file_root[: MAX_PATH_LENGTH - len(file_ext)] + file_ext

    return safe_filename


def upload_file(upload_dirname: str, file: UploadFile, filename: str):
    filename = sanitize_filename(filename)
    target_path = os.path.join(upload_dirname, filename)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    with open(target_path, "wb+") as upload_dir:
        shutil.copyfileobj(file.file, upload_dir)
    return {"status": "OK", "msg": f"Uploaded {filename}"}
