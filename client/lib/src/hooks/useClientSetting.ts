import { useState, useMemo } from "react"

import { VoiceChangerClientSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"

export type UseClientSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    audioContext: AudioContext | null
    defaultVoiceChangerClientSetting: VoiceChangerClientSetting
}

export type ClientSettingState = {
    clientSetting: VoiceChangerClientSetting;
    setServerUrl: (url: string) => void;
    updateClientSetting: (clientSetting: VoiceChangerClientSetting) => void

    start: () => Promise<void>
    stop: () => Promise<void>
    reloadClientSetting: () => Promise<void>
}

export const useClientSetting = (props: UseClientSettingProps): ClientSettingState => {
    const [clientSetting, setClientSetting] = useState<VoiceChangerClientSetting>(props.defaultVoiceChangerClientSetting)

    //////////////
    // 設定
    /////////////
    const updateClientSetting = useMemo(() => {
        return async (_clientSetting: VoiceChangerClientSetting) => {
            if (!props.voiceChangerClient) return
            for (let k in _clientSetting) {
                const cur_v = clientSetting[k as keyof VoiceChangerClientSetting]
                const new_v = _clientSetting[k as keyof VoiceChangerClientSetting]
                if (cur_v != new_v) {
                    setClientSetting(_clientSetting)
                    await props.voiceChangerClient.updateClientSetting(_clientSetting)
                    break
                }
            }
        }
    }, [props.voiceChangerClient, clientSetting])

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
            // props.voiceChangerClient.setServerUrl(setting.mmvcServerUrl, true)
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
        clientSetting,
        setServerUrl,
        updateClientSetting,

        start,
        stop,
        reloadClientSetting
    }
}