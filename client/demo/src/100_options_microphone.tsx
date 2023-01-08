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
    const clientState = useClient({
        audioContext: audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const serverSetting = useServerSetting({
        clientState
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
        responseTime: clientState.responseTime
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
        if (!clientState.clientInitialized) return
        clientState.setServerUrl(serverSetting.mmvcServerUrl)
    }, [clientState.clientInitialized, serverSetting.mmvcServerUrl])
    //// プロトコル変更
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.setProtocol(serverSetting.protocol)
    }, [clientState.clientInitialized, serverSetting.protocol])
    //// フレームワーク変更
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.framework, serverSetting.framework)
    }, [clientState.clientInitialized, serverSetting.framework])
    //// OnnxExecutionProvider変更
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.onnxExecutionProvider, serverSetting.onnxExecutionProvider)
    }, [clientState.clientInitialized, serverSetting.onnxExecutionProvider])

    // 102 DeviceSetting
    //// 入力情報の設定
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.changeInput(deviceSetting.audioInput, convertSetting.bufferSize, advancedSetting.vfForceDisabled)
    }, [clientState.clientInitialized, deviceSetting.audioInput, convertSetting.bufferSize, advancedSetting.vfForceDisabled])

    // 103 SpeakerSetting
    // 音声変換元、変換先の設定
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.srcId, speakerSetting.srcId)
    }, [clientState.clientInitialized, speakerSetting.srcId])
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.dstId, speakerSetting.dstId)
    }, [clientState.clientInitialized, speakerSetting.dstId])

    // 104 ConvertSetting
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.setInputChunkNum(convertSetting.inputChunkNum)
    }, [clientState.clientInitialized, convertSetting.inputChunkNum])
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.convertChunkNum, convertSetting.convertChunkNum)
    }, [clientState.clientInitialized, convertSetting.convertChunkNum])
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.gpu, convertSetting.gpu)
    }, [clientState.clientInitialized, convertSetting.gpu])
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.crossFadeOffsetRate, convertSetting.crossFadeOffsetRate)
    }, [clientState.clientInitialized, convertSetting.crossFadeOffsetRate])
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.updateSettings(ServerSettingKey.crossFadeEndRate, convertSetting.crossFadeEndRate)
    }, [clientState.clientInitialized, convertSetting.crossFadeEndRate])

    // 105 AdvancedSetting
    useEffect(() => {
        if (!clientState.clientInitialized) return
        clientState.setVoiceChangerMode(advancedSetting.voiceChangerMode)
    }, [clientState.clientInitialized, advancedSetting.voiceChangerMode])


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

