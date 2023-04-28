import torch
from transformers import HubertModel
from voice_changer.utils.VoiceChangerModel import AudioInOut


class RinnaHubertBase:
    def __init__(self):
        model = HubertModel.from_pretrained("rinna/japanese-hubert-base")
        model.eval()
        self.model = model

    def extract(self, audio: AudioInOut):
        return self.model(audio)
