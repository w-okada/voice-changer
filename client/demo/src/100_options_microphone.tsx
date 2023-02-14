import * as React from "react";
import { useEffect, useMemo, useState } from "react";
import { AUDIO_ELEMENT_FOR_PLAY_RESULT } from "./const";
import { useServerSettingArea } from "./101_server_setting";
import { useDeviceSetting } from "./102_device_setting";
import { useConvertSetting } from "./104_convert_setting";
import { useAdvancedSetting } from "./105_advanced_setting";
import { useSpeakerSetting } from "./103_speaker_setting";
import { useServerControl } from "./106_server_control";
import { useClient } from "@dannadori/voice-changer-client-js";

export const useMicrophoneOptions = () => {
    const [audioContext, setAudioContext] = useState<AudioContext | null>(null)

    const clientState = useClient({
        audioContext: audioContext,
        audioOutputElementId: AUDIO_ELEMENT_FOR_PLAY_RESULT
    })

    const serverSetting = useServerSettingArea({ clientState })
    const deviceSetting = useDeviceSetting(audioContext, { clientState })
    const speakerSetting = useSpeakerSetting({ clientState })
    const convertSetting = useConvertSetting({ clientState })
    const advancedSetting = useAdvancedSetting({ clientState })
    const serverControl = useServerControl({ clientState })

    const clearSetting = async () => {
        await clientState.clearSetting()
    }

    useEffect(() => {
        const createAudioContext = () => {
            const ctx = new AudioContext({
                sampleRate: 48000,
            })
            setAudioContext(ctx)
            document.removeEventListener('touchstart', createAudioContext);
            document.removeEventListener('mousedown', createAudioContext);
        }
        document.addEventListener('touchstart', createAudioContext);
        document.addEventListener('mousedown', createAudioContext);
    }, [])


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
        clearSetting
    }
}

