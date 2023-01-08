import * as React from "react";
import { useEffect, useMemo, useState } from "react";
import { AUDIO_ELEMENT_FOR_PLAY_RESULT } from "./const";
import { useServerSetting } from "./101_server_setting";
import { useDeviceSetting } from "./102_device_setting";
import { useConvertSetting } from "./104_convert_setting";
import { useAdvancedSetting } from "./105_advanced_setting";
import { useSpeakerSetting } from "./103_speaker_setting";
import { useClient } from "./hooks/useClient";
import { useServerControl } from "./106_server_control";
import { ServerSettingKey } from "@dannadori/voice-changer-client-js";



export const useMicrophoneOptions = () => {
    const [audioContext, setAudioContext] = useState<AudioContext | null>(null)
    const [loadModelFunc, setLoadModelFunc] = useState<() => Promise<void>>()
    const [uploadProgress, setUploadProgress] = useState<number>(0)
    const [isUploading, setIsUploading] = useState<boolean>(false)
    const clientState = useClient({
        audioContext: audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const serverSetting = useServerSetting({
        clientState,
        loadModelFunc,
        uploadProgress: uploadProgress,
        isUploading: isUploading
    })
    const deviceSetting = useDeviceSetting(audioContext)
    const speakerSetting = useSpeakerSetting()
    const convertSetting = useConvertSetting()
    const advancedSetting = useAdvancedSetting()


    const serverControl = useServerControl({
        convertStart: async () => { await clientState.start(serverSetting.mmvcServerUrl, serverSetting.protocol) },
        convertStop: async () => { clientState.stop() },
        getInfo: clientState.getInfo,
        volume: clientState.volume,
        bufferingTime: clientState.bufferingTime,
        responseTime: clientState.responseTime,

    })

    useEffect(() => {
        const createAudioContext = () => {
            const ctx = new AudioContext()
            setAudioContext(ctx)
            document.removeEventListener('touchstart', createAudioContext);
            document.removeEventListener('mousedown', createAudioContext);
        }
        document.addEventListener('touchstart', createAudioContext);
        document.addEventListener('mousedown', createAudioContext);
    }, [])

    // 101 ServerSetting
    //// サーバ変更
    useEffect(() => {
        clientState.setServerUrl(serverSetting.mmvcServerUrl)
    }, [serverSetting.mmvcServerUrl])
    //// プロトコル変更
    useEffect(() => {
        clientState.setProtocol(serverSetting.protocol)
    }, [serverSetting.protocol])
    //// フレームワーク変更
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.framework, serverSetting.framework)
    }, [serverSetting.framework])
    //// OnnxExecutionProvider変更
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.onnxExecutionProvider, serverSetting.onnxExecutionProvider)
    }, [serverSetting.onnxExecutionProvider])

    // 102 DeviceSetting
    //// 入力情報の設定
    useEffect(() => {
        clientState.changeInput(deviceSetting.audioInput, convertSetting.bufferSize, advancedSetting.vfForceDisabled)
    }, [deviceSetting.audioInput, convertSetting.bufferSize, advancedSetting.vfForceDisabled])

    // 103 SpeakerSetting
    // 音声変換元、変換先の設定
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.srcId, speakerSetting.srcId)
    }, [speakerSetting.srcId])
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.dstId, speakerSetting.dstId)
    }, [speakerSetting.dstId])

    // 104 ConvertSetting
    useEffect(() => {
        clientState.setInputChunkNum(convertSetting.inputChunkNum)
    }, [convertSetting.inputChunkNum])
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.convertChunkNum, convertSetting.convertChunkNum)
    }, [convertSetting.convertChunkNum])
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.gpu, convertSetting.gpu)
    }, [convertSetting.gpu])
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.crossFadeOffsetRate, convertSetting.crossFadeOffsetRate)
    }, [convertSetting.crossFadeOffsetRate])
    useEffect(() => {
        clientState.updateSettings(ServerSettingKey.crossFadeEndRate, convertSetting.crossFadeEndRate)
    }, [convertSetting.crossFadeEndRate])

    // 105 AdvancedSetting
    useEffect(() => {
        clientState.setVoiceChangerMode(advancedSetting.voiceChangerMode)
    }, [advancedSetting.voiceChangerMode])


    // Model Load
    useEffect(() => {
        const loadModel = () => {
            return async () => {
                if (!serverSetting.pyTorchModel && !serverSetting.onnxModel) {
                    alert("PyTorchモデルとONNXモデルのどちらか一つ以上指定する必要があります。")
                    return
                }
                if (!serverSetting.configFile) {
                    alert("Configファイルを指定する必要があります。")
                    return
                }
                setUploadProgress(0)
                setIsUploading(true)
                const models = [serverSetting.pyTorchModel, serverSetting.onnxModel].filter(x => { return x != null }) as File[]
                for (let i = 0; i < models.length; i++) {
                    const progRate = 1 / models.length
                    const progOffset = 100 * i * progRate
                    await clientState.uploadFile(models[i], (progress: number, end: boolean) => {
                        // console.log(progress * progRate + progOffset, end, progRate,)
                        setUploadProgress(progress * progRate + progOffset)
                    })
                }

                await clientState.uploadFile(serverSetting.configFile, (progress: number, end: boolean) => {
                    console.log(progress, end)
                })

                await clientState.loadModel(serverSetting.configFile, serverSetting.pyTorchModel, serverSetting.onnxModel)
                setUploadProgress(0)
                setIsUploading(false)
            }
        }
        setLoadModelFunc(loadModel)
    }, [serverSetting.configFile, serverSetting.pyTorchModel, serverSetting.onnxModel,
    serverSetting.framework, serverSetting.onnxExecutionProvider, speakerSetting.srcId, speakerSetting.dstId, convertSetting.gpu, convertSetting.crossFadeOffsetRate, convertSetting.crossFadeEndRate
    ])


    const voiceChangerSetting = useMemo(() => {
        return (
            <>
                <div className="body-row left-padding-1">
                    <div className="body-section-title">Virtual Microphone</div>
                </div>
                {serverControl.serverControl}
                {serverSetting.serverSetting}
                {deviceSetting.deviceSetting}
                {speakerSetting.speakerSetting}
                {convertSetting.convertSetting}
                {advancedSetting.advancedSetting}
            </>
        )
    }, [serverControl.serverControl,
    serverSetting.serverSetting,
    deviceSetting.deviceSetting,
    speakerSetting.speakerSetting,
    convertSetting.convertSetting,
    advancedSetting.advancedSetting])

    return {
        voiceChangerSetting,
    }
}

