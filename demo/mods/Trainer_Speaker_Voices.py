from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from trainer_mods.files import get_file_list
import os

def mod_get_speaker_voices(speaker:str):
        voices = get_file_list(f'MMVC_Trainer/dataset/textful/{speaker}/wav/*.wav')

        texts = get_file_list(f'MMVC_Trainer/dataset/textful/{speaker}/text/*.txt')

        readable_texts = get_file_list(f'MMVC_Trainer/dataset/textful/{speaker}/readable_text/*.txt')

        items = voices
        items.extend(texts)
        items.extend(readable_texts)
        items = [ os.path.splitext(os.path.basename(x))[0] for x in items]
        items = sorted(set(items))
        data = {
                "voices":items
                }
        json_compatible_item_data = jsonable_encoder(data)
        return JSONResponse(content=json_compatible_item_data)