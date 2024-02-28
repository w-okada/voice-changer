# from whisper_ppg.model import Whisper, ModelDimensions

# from whisper_ppg_custom._LightWhisper import LightWhisper
# from whisper_ppg_custom.Timer import Timer2
# from whisper_ppg_custom.whisper_ppg.audio import load_audio, pad_or_trim, log_mel_spectrogram
# from whisper_ppg_custom.whisper_ppg.model import Whisper, ModelDimensions
import torch

# import numpy as np
# from easy_vc_dev.utils.whisper.audio import load_audio, pad_or_trim
from .model import ModelDimensions, Whisper

# import onnx

# from onnxsim import simplify
# import json

# import onnxruntime


def load_model(path) -> Whisper:
    device = "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


# def pred_ppg(whisper: Whisper, wavPath: str, ppgPath: str):
#     print("pred")
#     # whisper = load_model("base.pt")  # "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
#     audio = load_audio(wavPath)
#     audln = audio.shape[0]
#     ppgln = audln // 320
#     print("audio.shape1", audio.shape, audio.shape[0] / 16000)
#     audio = pad_or_trim(audio)
#     audio = audio[:400000]
#     print("audio.shape2", audio.shape)
#     print(f"whisper.device {whisper.device}")
#     for i in range(5):
#         with Timer2("mainPorcess timer", True) as t:
#             mel = log_mel_spectrogram(audio).to(whisper.device)
#             with torch.no_grad():
#                 ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
#                 print("ppg.shape", ppg.shape)
#                 ppg = ppg[:ppgln,]
#                 print(ppg.shape)
#                 np.save(ppgPath, ppg, allow_pickle=False)
#             t.record("fin")
#     print("res", ppg)


# def pred_ppg_onnx(wavPath, ppgPath):
#     print("pred")
#     # whisper = load_model("base.pt")  # "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
#     whisper = load_model("tiny.pt")
#     audio = load_audio(wavPath)
#     # audln = audio.shape[0]
#     # ppgln = audln // 320
#     print("audio.shape1", audio.shape, audio.shape[0] / 16000)
#     audio = pad_or_trim(audio)
#     audio = audio[:1000]
#     print("audio.shape2", audio.shape)
#     print(f"whisper.device {whisper.device}")
#     onnx_session = onnxruntime.InferenceSession(
#         "wencoder_sim.onnx",
#         providers=["CPUExecutionProvider"],
#         provider_options=[
#             {
#                 "intra_op_num_threads": 8,
#                 "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
#                 "inter_op_num_threads": 8,
#             }
#         ],
#     )

#     for i in range(5):
#         with Timer2("mainPorcess timer", True) as t:
#             mel = log_mel_spectrogram(audio).to(whisper.device).unsqueeze(0)
#             onnx_res = onnx_session.run(
#                 ["ppg"],
#                 {
#                     "mel": mel.cpu().numpy(),
#                 },
#             )
#             t.record("fin")
#     print("onnx_res", onnx_res)


# def export_encoder(wavPath, ppgPath):
#     print("pred")
#     # whisper = load_model("base.pt")  # "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
#     whisper = load_model("tiny.pt")
#     audio = load_audio(wavPath)
#     # audln = audio.shape[0]
#     # ppgln = audln // 320
#     print("audio.shape1", audio.shape, audio.shape[0] / 16000)
#     audio = pad_or_trim(audio)
#     print("audio.shape2", audio.shape)
#     print(f"whisper.device {whisper.device}")

#     mel = log_mel_spectrogram(audio).to(whisper.device).unsqueeze(0)
#     input_names = ["mel"]
#     output_names = ["ppg"]

#     torch.onnx.export(
#         whisper.encoder,
#         (mel,),
#         "wencoder.onnx",
#         dynamic_axes={
#             "mel": [2],
#         },
#         do_constant_folding=False,
#         opset_version=17,
#         verbose=False,
#         input_names=input_names,
#         output_names=output_names,
#     )

#     metadata = {
#         "application": "VC_CLIENT",
#         "version": "2.1",
#     }
#     model_onnx2 = onnx.load("wencoder.onnx")
#     model_simp, check = simplify(model_onnx2)
#     meta = model_simp.metadata_props.add()
#     meta.key = "metadata"
#     meta.value = json.dumps(metadata)
#     onnx.save(model_simp, "wencoder_sim.onnx")


# def pred_ppg_onnx_w(wavPath, ppgPath):
#     print("pred")
#     audio = load_audio(wavPath)
#     print("audio.shape1", audio.shape, audio.shape[0] / 16000)
#     audio = pad_or_trim(audio)
#     print("audio.shape2", audio.shape)
#     onnx_session = onnxruntime.InferenceSession(
#         "wencoder_sim.onnx",
#         providers=["CPUExecutionProvider"],
#         provider_options=[
#             {
#                 "intra_op_num_threads": 8,
#                 "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
#                 "inter_op_num_threads": 8,
#             }
#         ],
#     )

#     for i in range(5):
#         with Timer2("mainPorcess timer", True) as t:
#             mel = log_mel_spectrogram(audio).to("cpu").unsqueeze(0)
#             # mel = mel[:, :, 1500:]
#             mel = mel[:, :, 2500:]
#             # mel[0, 79, 1499] = 0.1

#             print("x.shape", mel.shape)
#             onnx_res = onnx_session.run(
#                 ["ppg"],
#                 {
#                     "mel": mel.cpu().numpy(),
#                 },
#             )
#             t.record("fin")
#     print("onnx_res", onnx_res)


# def export_wrapped_encoder(wavPath, ppgPath):
#     print("pred")
#     whisper = LightWhisper("tiny.pt")
#     audio = load_audio(wavPath)
#     # audln = audio.shape[0]
#     # ppgln = audln // 320
#     print("audio.shape1", audio.shape, audio.shape[0] / 16000)
#     audio = pad_or_trim(audio)
#     print("audio.shape2", audio.shape)

#     mel = log_mel_spectrogram(audio).to("cpu").unsqueeze(0)
#     mel = mel[:, :, 1500:]
#     input_names = ["mel"]
#     output_names = ["ppg"]

#     torch.onnx.export(
#         whisper,
#         (mel,),
#         "wencoder.onnx",
#         dynamic_axes={
#             "mel": [2],
#         },
#         do_constant_folding=True,
#         opset_version=17,
#         verbose=False,
#         input_names=input_names,
#         output_names=output_names,
#     )

#     metadata = {
#         "application": "VC_CLIENT",
#         "version": "2.1",
#     }
#     model_onnx2 = onnx.load("wencoder.onnx")
#     model_simp, check = simplify(model_onnx2)
#     meta = model_simp.metadata_props.add()
#     meta.key = "metadata"
#     meta.value = json.dumps(metadata)
#     onnx.save(model_simp, "wencoder_sim.onnx")
