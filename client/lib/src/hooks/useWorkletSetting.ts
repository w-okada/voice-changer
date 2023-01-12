import { useState, useMemo, useEffect } from "react"
import { WorkletSetting, DefaultWorkletSetting } from "../const";
import { VoiceChangerClient } from "../VoiceChangerClient";

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


    //////////////
    // デフォルト設定
    /////////////
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const colab = params.get("colab")
        if (colab == "true") {
            setSetting({
                numTrancateTreshold: 300,
                volTrancateThreshold: 0.0005,
                volTrancateLength: 32,
            })
        } else {
            setSetting({
                numTrancateTreshold: 150,
                volTrancateThreshold: 0.0005,
                volTrancateLength: 32,
            })
        }
    }, [props.voiceChangerClient])

    return {
        setting,
        setSetting
    }
}