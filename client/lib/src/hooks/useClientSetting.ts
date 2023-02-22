import { useState, useMemo, useEffect } from "react"

import { VoiceChangerClientSetting, DefaultVoiceChangerClientSetting, INDEXEDDB_KEY_CLIENT } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"

export type UseClientSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    audioContext: AudioContext | null
}

export type ClientSettingState = {
    clientSetting: VoiceChangerClientSetting;
    clearSetting: () => Promise<void>
    setServerUrl: (url: string) => void;
    updateClientSetting: (clientSetting: VoiceChangerClientSetting) => void

    start: () => Promise<void>
    stop: () => Promise<void>
    reloadClientSetting: () => Promise<void>
}

export const useClientSetting = (props: UseClientSettingProps): ClientSettingState => {
    const [clientSetting, setClientSetting] = useState<VoiceChangerClientSetting>(DefaultVoiceChangerClientSetting)
    const { setItem, getItem, removeItem } = useIndexedDB()

    // 初期化 その１ DBから取得
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_CLIENT) as VoiceChangerClientSetting
            if (!setting) {
                return
            }

            console.log("[ClientSetting] Load Setting from db", setting)
            if (setting.audioInput == "null") {
                setting.audioInput = null
            }
            if (setting) {
                setClientSetting({ ...setting })
            }
        }
        loadCache()
    }, [])
    // 初期化 その２ クライアントに設定
    useEffect(() => {
        if (!props.voiceChangerClient) return
        props.voiceChangerClient.updateClientSetting(clientSetting)
    }, [props.voiceChangerClient])


    const storeSetting = async (setting: VoiceChangerClientSetting) => {
        const storeData = { ...setting }
        if (typeof storeData.audioInput != "string") {
            storeData.audioInput = null
        }
        setItem(INDEXEDDB_KEY_CLIENT, storeData)
        setClientSetting(setting)
    }

    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_CLIENT)
    }

    //////////////
    // 設定
    /////////////
    const updateClientSetting = useMemo(() => {
        return (_clientSetting: VoiceChangerClientSetting) => {
            if (!props.voiceChangerClient) return
            for (let k in _clientSetting) {
                const cur_v = clientSetting[k as keyof VoiceChangerClientSetting]
                const new_v = _clientSetting[k as keyof VoiceChangerClientSetting]
                if (cur_v != new_v) {
                    storeSetting(_clientSetting)
                    props.voiceChangerClient.updateClientSetting(_clientSetting)
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
            props.voiceChangerClient.start()
        }
    }, [props.voiceChangerClient])
    // (2) stop
    const stop = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.stop()
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
        clearSetting,
        setServerUrl,
        updateClientSetting,

        start,
        stop,
        reloadClientSetting
    }
}