import { DefaultWorkletSetting, VoiceChangerClient, WorkletSetting } from "@dannadori/voice-changer-client-js"
import { useState, useMemo } from "react"

export type UseWorkletSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type WorkletSettingState = {
    setting: WorkletSetting;
    setSetting: (setting: WorkletSetting) => void;
}

export const useWorkletSetting = (props: UseWorkletSettingProps): WorkletSettingState => {
    const [setting, _setSetting] = useState<WorkletSetting>(DefaultWorkletSetting)


    const setSetting = useMemo(() => {
        return (setting: WorkletSetting) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.configureWorklet(setting)
            _setSetting(setting)
        }
    }, [props.voiceChangerClient])

    return {
        setting,
        setSetting
    }
}