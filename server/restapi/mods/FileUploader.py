import os, shutil
from fastapi import UploadFile

# UPLOAD_DIR = "model_upload_dir"

def upload_file(upload_dirname:str, file:UploadFile, filename: str):
    if file and filename:
        fileobj = file.file
        upload_dir = open(os.path.join(upload_dirname, filename),'wb+')
        shutil.copyfileobj(fileobj, upload_dir)
        upload_dir.close()
        return {"uploaded files": f"{filename} "}
    return {"Error": "uploaded file is not found."}

def concat_file_chunks(upload_dirname:str, filename:str, chunkNum:int, dest_dirname:str):
    target_file_name = os.path.join(dest_dirname, filename)
    if os.path.exists(target_file_name):
        os.unlink(target_file_name)
    with open(target_file_name, "ab") as target_file:
        for i in range(chunkNum):
            chunkName = f"{filename}_{i}"
            chunk_file_path = os.path.join(upload_dirname, chunkName)
            stored_chunk_file = open(chunk_file_path, 'rb')
            target_file.write(stored_chunk_file.read())
            stored_chunk_file.close()
            os.unlink(chunk_file_path)
        target_file.close()
    return target_file_name

