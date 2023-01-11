import shutil
from restapi.mods.Trainer_MultiSpeakerSetting import MULTI_SPEAKER_SETTING_PATH

def mod_delete_speaker(speaker:str):
        shutil.rmtree(f"MMVC_Trainer/dataset/textful/{speaker}")

        with open(MULTI_SPEAKER_SETTING_PATH, "r") as f:
            setting = f.readlines()

        filtered = filter(lambda x: x.startswith(f"{speaker}|")==False, setting)
        with open(MULTI_SPEAKER_SETTING_PATH, "w") as f:
            f.writelines(list(filtered))
            f.flush()
        f.close()
        return {"Speaker deleted": f"{speaker}"}