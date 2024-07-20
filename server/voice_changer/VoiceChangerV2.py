from typing import Any, Union

from const import TMP_DIR
from torch.functional import F
import torch
import os
import numpy as np
import logging

from voice_changer.IORecorder import IORecorder
from voice_changer.VoiceChangerSettings import VoiceChangerSettings
from voice_changer.utils.Timer import Timer2
from voice_changer.utils.VoiceChangerIF import VoiceChangerIF
from voice_changer.utils.VoiceChangerModel import AudioInOutFloat, VoiceChangerModel
from Exceptions import (
    VoiceChangerIsNotSelectedException,
)
from settings import ServerSettings
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")
logger = logging.getLogger(__name__)


class VoiceChangerV2(VoiceChangerIF):
    def __init__(self, params: ServerSettings, settings: VoiceChangerSettings):
        # 初期化
        self.settings = settings

        self.block_frame = self.settings.serverReadChunkSize * 128
        self.crossfade_frame = int(self.settings.crossFadeOverlapSize * self.settings.inputSampleRate)
        self.extra_frame = int(self.settings.extraConvertSize * self.settings.inputSampleRate)
        self.sola_search_frame = self.settings.inputSampleRate // 100

        self.voiceChangerModel: VoiceChangerModel | None = None
        self.params = params
        self.device_manager = DeviceManager.get_instance()
        self.sola_buffer: torch.Tensor | None = None
        self.ioRecorder: IORecorder | None = None

        logger.info(f"VoiceChangerV2 Initialized")
        np.set_printoptions(threshold=10000)


    def set_model(self, model: VoiceChangerModel):
        self.voiceChangerModel = model
        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
        self._generate_strength()

    def set_input_sample_rate(self):
        self.extra_frame = int(self.settings.extraConvertSize * self.settings.inputSampleRate)
        self.crossfade_frame = int(self.settings.crossFadeOverlapSize * self.settings.inputSampleRate)
        self.sola_search_frame = self.settings.inputSampleRate // 100
        self._generate_strength()

        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)

    def set_output_sample_rate(self):
        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)

    def get_info(self):
        if self.voiceChangerModel is not None:
            return self.voiceChangerModel.get_info()
        return {}

    def update_settings(self, key: str, val: Any, old_val: Any):
        if self.voiceChangerModel is None:
            logger.warn("Voice Changer model is not selected.")
            return

        if key == "serverReadChunkSize":
            self.block_frame = self.settings.serverReadChunkSize * 128
        elif key == 'gpu':
            # When changing GPU, need to re-allocate fade-in/fade-out buffers on different device
            self._generate_strength()
        elif key == "recordIO":
            if val:
                self.ioRecorder = IORecorder(
                    STREAM_INPUT_FILE,
                    STREAM_OUTPUT_FILE,
                    self.settings.inputSampleRate,
                    self.settings.outputSampleRate,
                    # 16000,
                )
                logger.info(f"-------------------------- - - - {self.settings.inputSampleRate}, {self.settings.outputSampleRate}")
            else:
                self.ioRecorder.close()
        elif key == "inputSampleRate":
            self.set_input_sample_rate()
        elif key == "outputSampleRate":
            self.set_output_sample_rate()
        elif key == 'extraConvertSize':
            self.extra_frame = int(val * self.settings.inputSampleRate)
        elif key == 'crossFadeOverlapSize':
            self.crossfade_frame = int(val * self.settings.inputSampleRate)
            self._generate_strength()

        self.voiceChangerModel.update_settings(key, val, old_val)
        if key in {'gpu', 'serverReadChunkSize', 'extraConvertSize', 'crossFadeOverlapSize', 'silenceFront', 'forceFp32'}:
            self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)


    def _generate_strength(self):
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.crossfade_frame,
                    device=self.device_manager.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window

        # ひとつ前の結果とサイズが変わるため、記録は消去する。
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device_manager.device, dtype=torch.float32)
        logger.info(f'Allocated SOLA buffer size: {self.crossfade_frame}')

    def get_processing_sampling_rate(self):
        if self.voiceChangerModel is None:
            return 0
        return self.voiceChangerModel.get_processing_sampling_rate()

    @torch.no_grad()
    def on_request(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, list[Union[int, float]]]:
        if self.voiceChangerModel is None:
            raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")

        with Timer2("main-process", True) as t:
            block_size = audio_in.shape[0]

            audio = self.voiceChangerModel.inference(audio_in)

            if audio is None:
                # In case there's an actual silence - send full block with zeros
                return np.zeros(block_size, dtype=np.float32), [0, 0, 0]

            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
            conv_input = audio[
                None, None, : self.crossfade_frame + self.sola_search_frame
            ]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input ** 2,
                    torch.ones(1, 1, self.crossfade_frame, device=self.device_manager.device),
                )
                + 1e-8
            )
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

            audio = audio[sola_offset:]
            audio[: self.crossfade_frame] *= self.fade_in_window
            audio[: self.crossfade_frame] += (
                self.sola_buffer * self.fade_out_window
            )

            self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]

            result: np.ndarray = audio[: block_size].detach().cpu().numpy()

        mainprocess_time = t.secs

        # 後処理
        if self.settings.recordIO:
            self.ioRecorder.writeInput((audio_in * 32767).astype(np.int16).tobytes())
            self.ioRecorder.writeOutput((result * 32767).astype(np.int16).tobytes())

        return result, [0, mainprocess_time, 0]

    @torch.no_grad()
    def export2onnx(self):
        return self.voiceChangerModel.export2onnx()
