import * as React from "react";
import { useEffect, useMemo, useState } from "react";
import { AUDIO_ELEMENT_FOR_PLAY_RESULT, CHROME_EXTENSION } from "./const";
import { useServerSetting } from "./101_server_setting";
import { useDeviceSetting } from "./102_device_setting";
import { useConvertSetting } from "./104_convert_setting";
import { useAdvancedSetting } from "./105_advanced_setting";
import { useSpeakerSetting } from "./103_speaker_setting";
import { useClient } from "./hooks/useClient";
import { useServerControl } from "./106_server_control";



export const useMicrophoneOptions = () => {
    const [audioContext, setAudioContext] = useState<AudioContext | null>(null)
    const clientState = useClient({
        audioContext: audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const serverSetting = useServerSetting({
        uploadFile: clientState.uploadFile,
        changeOnnxExcecutionProvider: clientState.changeOnnxExcecutionProvider
    })
    const deviceSetting = useDeviceSetting(audioContext)
    const speakerSetting = useSpeakerSetting()
    const convertSetting = useConvertSetting()
    const advancedSetting = useAdvancedSetting()


    const serverControl = useServerControl({
        convertStart: async () => { await clientState.start(serverSetting.mmvcServerUrl, serverSetting.protocol) },
        convertStop: async () => { clientState.stop() },
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



    useEffect(() => {
        console.log("input Cahngaga!")
        clientState.changeInput(deviceSetting.audioInput, convertSetting.bufferSize, advancedSetting.vfForceDisabled)
    }, [clientState.clientInitialized, deviceSetting.audioInput, convertSetting.bufferSize, advancedSetting.vfForceDisabled])



    // // const [options, setOptions] = useState<MicrophoneOptionsState>(InitMicrophoneOptionsState)
    // const [params, setParams] = useState<VoiceChangerRequestParamas>(DefaultVoiceChangerRequestParamas)
    // const [options, setOptions] = useState<VoiceChangerOptions>(DefaultVoiceChangerOptions)
    // const [isStarted, setIsStarted] = useState<boolean>(false)


    // useEffect(() => {
    //     const storeOptions = async () => {
    //         if (CHROME_EXTENSION) {
    //             // @ts-ignore
    //             await chrome.storage.local.set({ microphoneOptions: options })
    //         }
    //     }
    //     storeOptions()
    // }, [options]) // loadより前に持ってくるとstorage内が初期化されるのでだめかも。（要検証）




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

