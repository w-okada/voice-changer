import { useState, useMemo, useEffect } from "react"

import { VoiceChangerClientSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"

export type UseClientSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    voiceChangerClientSetting: VoiceChangerClientSetting
}

export type ClientSettingState = {

    setServerUrl: (url: string) => void;
    start: () => Promise<void>
    stop: () => Promise<void>
    reloadClientSetting: () => Promise<void>
}

export const useClientSetting = (props: UseClientSettingProps): ClientSettingState => {
    // 更新比較用
    const [voiceChangerClientSetting, setVoiceChangerClientSetting] = useState<VoiceChangerClientSetting>(props.voiceChangerClientSetting)

    useEffect(() => {
        const update = async () => {
            if (!props.voiceChangerClient) return
            for (let k in props.voiceChangerClientSetting) {
                const cur_v = voiceChangerClientSetting[k as keyof VoiceChangerClientSetting]
                const new_v = props.voiceChangerClientSetting[k as keyof VoiceChangerClientSetting]
                if (cur_v != new_v) {
                    setVoiceChangerClientSetting(props.voiceChangerClientSetting)
                    await props.voiceChangerClient.updateClientSetting(props.voiceChangerClientSetting)
                    break
                }
            }
        }
        update()
    }, [props.voiceChangerClient, props.voiceChangerClientSetting])


    const setServerUrl = useMemo(() => {
        return (url: string) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setServerUrl(url, true)
        }
    }, [props.voiceChangerClient])


    //////////////
    // 操作
    /////////////
    // (1) start
    const start = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            await props.voiceChangerClient.start()
        }
    }, [props.voiceChangerClient])
    // (2) stop
    const stop = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            await props.voiceChangerClient.stop()
        }
    }, [props.voiceChangerClient])
    const reloadClientSetting = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            await props.voiceChangerClient.getClientSettings()
        }
    }, [props.voiceChangerClient])

    return {
        setServerUrl,

        start,
        stop,
        reloadClientSetting
    }
}