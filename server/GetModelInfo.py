
from voice_changer.RVC.RVCModelSlotGenerator import RVCModelSlotGenerator
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager
from voice_changer.utils.LoadModelParams import LoadModelParamFile, LoadModelParams
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


if __name__ == "__main__":
    print("main")
    voiceChangerParams = VoiceChangerParams(
        model_dir="./model_dir/0",  # <----   Change HERE
        content_vec_500="",
        content_vec_500_onnx="",
        content_vec_500_onnx_on="",
        hubert_base="",
        hubert_base_jp="",
        hubert_soft="",
        nsf_hifigan="",
        crepe_onnx_full="",
        crepe_onnx_tiny="",
        rmvpe="",
        rmvpe_onnx="",
        sample_mode=""
    )
    vcparams = VoiceChangerParamsManager.get_instance()
    vcparams.setParams(voiceChangerParams)

    file = LoadModelParamFile(
        name="tsukuyomi_v2_40k_e100_simple.onnx",  # <----   Change HERE
        kind="rvcModel",
        dir="",
    )

    loadParam = LoadModelParams(
        voiceChangerType="RVC",
        files=[file],
        slot="",
        isSampleMode=False,
        sampleId="",
        params={},
    )
    slotInfo = RVCModelSlotGenerator.loadModel(loadParam)
    print(slotInfo.samplingRate)
