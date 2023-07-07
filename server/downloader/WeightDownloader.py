import os
from concurrent.futures import ThreadPoolExecutor

from downloader.Downloader import download
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from Exceptions import WeightDownladException


def downloadWeight(voiceChangerParams: VoiceChangerParams):
    content_vec_500_onnx = voiceChangerParams.content_vec_500_onnx
    hubert_base = voiceChangerParams.hubert_base
    hubert_base_jp = voiceChangerParams.hubert_base_jp
    hubert_soft = voiceChangerParams.hubert_soft
    nsf_hifigan = voiceChangerParams.nsf_hifigan
    crepe_onnx_full = voiceChangerParams.crepe_onnx_full
    crepe_onnx_tiny = voiceChangerParams.crepe_onnx_tiny

    # file exists check (currently only for rvc)
    downloadParams = []
    if os.path.exists(hubert_base) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
                "saveTo": hubert_base,
                "position": 0,
            }
        )
    if os.path.exists(hubert_base_jp) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
                "saveTo": hubert_base_jp,
                "position": 1,
            }
        )
    if os.path.exists(hubert_soft) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
                "saveTo": hubert_soft,
                "position": 2,
            }
        )
    if os.path.exists(nsf_hifigan) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin",
                "saveTo": nsf_hifigan,
                "position": 3,
            }
        )
    nsf_hifigan_config = os.path.join(os.path.dirname(nsf_hifigan), "config.json")

    if os.path.exists(nsf_hifigan_config) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json",
                "saveTo": nsf_hifigan_config,
                "position": 4,
            }
        )

    if os.path.exists(crepe_onnx_full) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx",
                "saveTo": crepe_onnx_full,
                "position": 5,
            }
        )
    if os.path.exists(crepe_onnx_tiny) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx",
                "saveTo": crepe_onnx_tiny,
                "position": 6,
            }
        )

    if os.path.exists(content_vec_500_onnx) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
                "saveTo": content_vec_500_onnx,
                "position": 7,
            }
        )

    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    if os.path.exists(hubert_base) is False or os.path.exists(hubert_base_jp) is False or os.path.exists(hubert_soft) is False or os.path.exists(nsf_hifigan) is False or os.path.exists(nsf_hifigan_config) is False:
        raise WeightDownladException()
