from typing import Any, Union

from const import TMP_DIR
from torch.functional import F
import torch
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from mods.log_control import VoiceChangaerLogger

from voice_changer.IORecorder import IORecorder

from voice_changer.utils.Timer import Timer2
from voice_changer.utils.VoiceChangerIF import VoiceChangerIF
from voice_changer.utils.VoiceChangerModel import AudioInOutFloat, VoiceChangerModel
from Exceptions import (
    DeviceChangingException,
    HalfPrecisionChangingException,
    NoModeLoadedException,
    PipelineNotInitializedException,
    VoiceChangerIsNotSelectedException,
)
from settings import ServerSettings
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")
logger = VoiceChangaerLogger.get_instance().getLogger()


@dataclass
class VoiceChangerV2Settings:
    inputSampleRate: int = 48000  # 48000 or 24000
    outputSampleRate: int = 48000  # 48000 or 24000

    crossFadeOverlapSize: float = 0.10
    serverReadChunkSize: int = 192
    extraConvertSize: float = 0.5
    gpu: int = -1
    forceFp32: int = 0 # 0:off, 1:on

    recordIO: int = 0  # 0:off, 1:on

    performance: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # ↓mutableな物だけ列挙
    intData: list[str] = field(
        default_factory=lambda: [
            "inputSampleRate",
            "outputSampleRate",
            "recordIO",
            "serverReadChunkSize",
            "gpu",
            "forceFp32",
        ]
    )
    strData: list[str] = field(default_factory=lambda: [])
    floatData: list[str] = field(
        default_factory=lambda: [
            "extraConvertSize",
            "protect",
            "crossFadeOverlapSize",
        ]
    )


class VoiceChangerV2(VoiceChangerIF):
    def __init__(self, params: ServerSettings):
        # 初期化
        self.settings = VoiceChangerV2Settings()

        self.block_frame = self.settings.serverReadChunkSize * 128
        self.crossfade_frame = int(self.settings.crossFadeOverlapSize * self.settings.inputSampleRate)
        self.extra_frame = int(self.settings.extraConvertSize * self.settings.inputSampleRate)
        self.sola_search_frame = int(0.012 * self.settings.inputSampleRate)

        self.processing_sampling_rate = 0

        self.voiceChangerModel: VoiceChangerModel | None = None
        self.params = params
        self.device_manager = DeviceManager.get_instance()
        self.noCrossFade = False # TODO: For the future, if other voice changing algos won't require crossfade
        self.sola_buffer: torch.Tensor | None = None
        self.ioRecorder: IORecorder | None = None

        logger.info(f"VoiceChangerV2 Initialized")
        np.set_printoptions(threshold=10000)


    def setModel(self, model: VoiceChangerModel):
        self.voiceChangerModel = model
        self.processing_sampling_rate = self.voiceChangerModel.get_processing_sampling_rate()
        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
        self._generate_strength()

    def setInputSampleRate(self, sr: int):
        self.settings.inputSampleRate = sr

        self.extra_frame = int(self.settings.extraConvertSize * sr)
        self.crossfade_frame = int(self.settings.crossFadeOverlapSize * sr)
        self.sola_search_frame = int(0.012 * sr)
        self._generate_strength()

        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)

    def setOutputSampleRate(self, sr: int):
        self.settings.outputSampleRate = sr
        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)

    def get_info(self):
        data = asdict(self.settings)
        if self.voiceChangerModel is not None:
            data.update(self.voiceChangerModel.get_info())
        return data

    def get_performance(self):
        return self.settings.performance

    def update_settings(self, key: str, val: Any):
        if self.voiceChangerModel is None:
            logger.warn("[Voice Changer] Voice Changer is not selected.")
            return self.get_info()

        if key == "serverAudioStated" and val == 0:
            self.settings.inputSampleRate = 48000
            self.settings.outputSampleRate = 48000
            self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)

        if key in self.settings.intData:
            val = int(val)
            setattr(self.settings, key, val)
            if key == "serverReadChunkSize":
                self.block_frame = self.settings.serverReadChunkSize * 128
            elif key == 'forceFp32':
                self.device_manager.set_force_fp32(val)
            elif key == 'gpu':
                self.device_manager.set_device(val)
                # When changing GPU, need to re-allocate fade-in/fade-out buffers on different device
                self._generate_strength()
            elif key == "recordIO" and val == 1:
                if self.ioRecorder is not None:
                    self.ioRecorder.close()
                self.ioRecorder = IORecorder(
                    STREAM_INPUT_FILE,
                    STREAM_OUTPUT_FILE,
                    self.settings.inputSampleRate,
                    self.settings.outputSampleRate,
                    # 16000,
                )
                print(f"-------------------------- - - - {self.settings.inputSampleRate}, {self.settings.outputSampleRate}")
            elif key == "recordIO" and val == 0:
                if self.ioRecorder is not None:
                    self.ioRecorder.close()
                pass
            elif key == "recordIO" and val == 2:
                if self.ioRecorder is not None:
                    self.ioRecorder.close()
            elif key in {"inputSampleRate", "outputSampleRate"}:
                self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        elif key in self.settings.floatData:
            val = float(val)
            setattr(self.settings, key, val)
            if key == 'extraConvertSize':
                self.extra_frame = int(val * self.settings.inputSampleRate)
            elif key == 'crossFadeOverlapSize':
                self.crossfade_frame = int(val * self.settings.inputSampleRate)
                self._generate_strength()

        if self.voiceChangerModel is not None:
            self.voiceChangerModel.update_settings(key, val)
            if key in {'gpu', 'serverReadChunkSize', 'extraConvertSize', 'crossFadeOverlapSize', 'rvcQuality', 'silenceFront', 'forceFp32'}:
                self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)

        return self.get_info()

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
        logger.info(f'[Voice Changer] Allocated sola buffer size: {self.sola_buffer.shape}')

    def get_processing_sampling_rate(self):
        if self.voiceChangerModel is None:
            return 0
        return self.voiceChangerModel.get_processing_sampling_rate()

    #  audio_in: tuple of short
    def on_request(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, list[Union[int, float]]]:
        try:
            if self.voiceChangerModel is None:
                raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")

            with Timer2("main-process", True) as t:
                block_size = audio_in.shape[0]

                audio = self.voiceChangerModel.inference(audio_in)

                if audio is None:
                    return np.zeros(block_size, dtype=np.float32), [0, 0, 0]

                if not self.noCrossFade:
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

                    end = block_size + self.crossfade_frame
                    if audio.shape[0] >= end:
                        self.sola_buffer[:] = audio[block_size : end]
                    else:
                        # No idea why but either new RVC code seems to produce smaller audio
                        # or SOLA offset calculation is a bit off. This fixes audio "popping" when fading chunks.
                        offset = end - audio.shape[0]
                        self.sola_buffer[-offset :] = torch.zeros(offset, device=self.device_manager.device, dtype=torch.float32)
                        self.sola_buffer[: -offset] = audio[block_size :]

                result: np.ndarray = audio[: block_size].detach().cpu().numpy()

            mainprocess_time = t.secs

            # 後処理
            with Timer2("post-process", True) as t:
                # print(f" Output data size of {result.shape[0]}/{self.processing_sampling_rate}hz {result.shape[0]}/{self.settings.outputSampleRate}hz")

                if self.settings.recordIO == 1:
                    self.ioRecorder.writeInput((audio_in * 32767).astype(np.int16))
                    self.ioRecorder.writeOutput((result * 32767).astype(np.int16).tobytes())

            postprocess_time = t.secs

            # print(f" [fin] Input/Output size:{audio_in.shape[0]},{result.shape[0]}")
            perf = [0, mainprocess_time, postprocess_time]

            return result, perf

        except NoModeLoadedException as e:
            logger.warn(f"[Voice Changer] [Exception], {e}")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except HalfPrecisionChangingException:
            logger.warn("[Voice Changer] Switching model configuration....")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except DeviceChangingException as e:
            logger.warn(f"[Voice Changer] embedder: {e}")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except VoiceChangerIsNotSelectedException:
            logger.warn("[Voice Changer] Voice Changer is not selected. Wait a bit and if there is no improvement, please re-select vc.")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except PipelineNotInitializedException:
            logger.warn("[Voice Changer] Pipeline is not initialized.")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except Exception as e:
            logger.warn(f"[Voice Changer] VC PROCESSING EXCEPTION!!! {e}")
            logger.exception(e)
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]

    def export2onnx(self):
        return self.voiceChangerModel.export2onnx()

        ##############

    def merge_models(self, request: str):
        if self.voiceChangerModel is None:
            logger.info("[Voice Changer] Voice Changer is not selected.")
            return
        self.voiceChangerModel.merge_models(request)
        return self.get_info()
