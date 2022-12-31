import subprocess,os
from trainer_mods.files import get_file_list
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

LOG_DIR = "info"
train_proc = None

SUCCESS = 0
ERROR = -1
### Submodule for Pre train
def sync_exec(cmd:str, log_path:str, cwd=None):
    shortCmdStr = cmd[:20]
    try:
        with open(log_path, 'w') as log_file:
            if cwd == None:
                proc = subprocess.run(cmd, shell=True, text=True, stdout=log_file, stderr=log_file)
            else:
                proc = subprocess.run(cmd, shell=True, text=True, stdout=log_file, stderr=log_file, cwd=cwd)
            print(f"{shortCmdStr} returncode:{proc.returncode}")
            if proc.returncode != 0:
                print(f"{shortCmdStr} exception:")
                return (ERROR, f"returncode:{proc.returncode}")
    except Exception as e:
        print(f"{shortCmdStr} exception:", str(e))
        return (ERROR, str(e))
    return (SUCCESS, "success")

def sync_exec_with_stdout(cmd:str, log_path:str):
    shortCmdStr = cmd[:20]
    try:
        with open(log_path, 'w') as log_file:
            proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE,
            stderr=log_file, cwd="MMVC_Trainer")
            print(f"STDOUT{shortCmdStr}",proc.stdout)
    except Exception as e:
        print(f"{shortCmdStr} exception:", str(e))
        return (ERROR, str(e))
    return (SUCCESS, proc.stdout)


def create_dataset():
    cmd = "python3 create_dataset_jtalk.py -f train_config -s 24000 -m dataset/multi_speaker_correspondence.txt"
    log_file = os.path.join(LOG_DIR, "log_create_dataset_jtalk.txt")
    res = sync_exec(cmd, log_file, "MMVC_Trainer")
    return res

def set_batch_size(batch:int):
    cmd = "sed -i 's/\"batch_size\": [0-9]*/\"batch_size\": " + str(batch) + "/' MMVC_Trainer/configs/baseconfig.json"
    log_file = os.path.join(LOG_DIR, "log_set_batch_size.txt")
    res = sync_exec(cmd, log_file)
    return res

def set_dummy_device_count():
    cmd = 'sed -ie "s/torch.cuda.device_count()/1/" MMVC_Trainer/train_ms.py'
    log_file = os.path.join(LOG_DIR, "log_set_dummy_device_count.txt")
    res = sync_exec(cmd, log_file)
    return res

### Submodule for Train 
def exec_training(enable_finetuning:bool, GModel:str, DModel:str):
    global train_proc
    log_file = os.path.join(LOG_DIR, "training.txt")

    # トレーニング開始確認(二重起動回避)
    if train_proc != None:
        status = train_proc.poll()
        if status != None:
            print("Training have ended.", status)
            train_proc = None
        else:
            print("Training have stated.")
            return (ERROR, "Training have started")

    try:
        with open(log_file, 'w') as log_file:
            if enable_finetuning == True:
                GModelPath = os.path.join("logs", GModel) # 実行時にcwdを指定しているのでフォルダはlogsでよい。
                DModelPath = os.path.join("logs", DModel)
                cmd = f'python3 train_ms.py -c configs/train_config.json -m ./ -fg {GModelPath} -fd {DModelPath}'
            else:
                cmd = 'python3 train_ms.py -c configs/train_config.json -m ./'
            print("exec:",cmd)
            train_proc = subprocess.Popen("exec "+cmd, shell=True, text=True, stdout=log_file, stderr=log_file, cwd="MMVC_Trainer")
            print("Training stated")
            print(f"returncode:{train_proc.returncode}")
    except Exception as e:
        print("start training exception:", str(e))
        return (ERROR,  str(e))

    return (SUCCESS, "success")

def stop_training():
    global train_proc
    if train_proc == None:
        print("Training have not stated.")
        return (ERROR, "Training have not stated.")

    status = train_proc.poll()
    if status != None:
        print("Training have already ended.", status)
        train_proc = None
        return (ERROR, "Training have already ended. " + status)
    else:
        train_proc.kill()
        print("Training have stoped.")
        return (SUCCESS, "success")

### Main
def mod_post_pre_training(batch:int):
    res = set_batch_size(batch)
    if res[0] == ERROR:
        return {"result":"failed", "detail": f"Preprocess(set_batch_size) failed. {res[1]}"}

    res = set_dummy_device_count()
    if res[0] == ERROR:
        return {"result":"failed", "detail": f"Preprocess(set_dummy_device_count) failed. {res[1]}"}

    res = create_dataset()
    if res[0] == ERROR:
        return {"result":"failed", "detail": f"Preprocess failed(create_dataset). {res[1]}"}

    return {"result":"success", "detail": f"Preprocess succeeded. {res[1]}"}


def mod_post_start_training(enable_finetuning:str, GModel:str, DModel:str):
    print("START_TRAINING:::::::", enable_finetuning, GModel, DModel)
    res = exec_training(enable_finetuning, GModel, DModel)
    if res[0] == ERROR:
        return {"result":"failed", "detail": f"Start training failed. {res[1]}"}

    return {"result":"success", "detail": f"Start training succeeded. {res[1]}"}

def mod_post_stop_training():
    res = stop_training()
    if res[0] == ERROR:
        return {"result":"failed", "detail": f"Stop training failed. {res[1]}"}

    return {"result":"success", "detail": f"Stop training succeeded. {res[1]}"}

### DEBUG
def mod_get_related_files():
    files = get_file_list(os.path.join(LOG_DIR,"*"))
    files.extend([
        "MMVC_Trainer/dataset/multi_speaker_correspondence.txt",
        "MMVC_Trainer/train_ms.py",
    ])
    files.extend(
        get_file_list("MMVC_Trainer/configs/*")
    )

    res = []
    for f in files:
        size = os.path.getsize(f)
        data = ""
        if size < 1024*1024:
            with open(f, "r") as input:
                data = input.read()
    
        res.append({
            "name":f,
            "size":size,
            "data":data
        })

    json_compatible_item_data = jsonable_encoder(res)
    return JSONResponse(content=json_compatible_item_data)

def mod_get_tail_training_log(num:int):
    training_log_file = os.path.join(LOG_DIR, "training.txt")
    res = sync_exec(f"cat {training_log_file} | sed -e 's/.*\r//' > /tmp/out","/dev/null")
    cmd = f'tail -n {num} /tmp/out'
    res = sync_exec_with_stdout(cmd, "/dev/null")
    if res[0] == ERROR:
        return {"result":"failed", "detail": f"Tail training log failed. {res[1]}"}
    return {"result":"success", "detail":res[1]}
