import { useState, useMemo, useEffect } from "react"

import { INDEXEDDB_KEY_CLIENT, INDEXEDDB_KEY_STREAMER, AudioStreamerSetting, DefaultAudioStreamerSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"

export type UseAudioStreamerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type AudioStreamerSettingState = {
    audioStreamerSetting: AudioStreamerSetting;
    clearSetting: () => Promise<void>
    setSetting: (setting: AudioStreamerSetting) => void

}

export const useAudioStreamerSetting = (props: UseAudioStreamerSettingProps): AudioStreamerSettingState => {
    const [audioStreamerSetting, _setAudioStreamerSetting] = useState<AudioStreamerSetting>(DefaultAudioStreamerSetting)
    const { setItem, getItem, removeItem } = useIndexedDB()

    // 初期化 その１ DBから取得
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_STREAMER) as AudioStreamerSetting
            if (setting) {
                _setAudioStreamerSetting(setting)
            }
        }
        loadCache()
    }, [])

    // 初期化 その２ クライアントに設定
    useEffect(() => {
        if (!props.voiceChangerClient) return
        props.voiceChangerClient.setServerUrl(audioStreamerSetting.serverUrl)
        props.voiceChangerClient.updateAudioStreamerSetting(audioStreamerSetting)
    }, [props.voiceChangerClient])



    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_STREAMER)
    }

    //////////////
    // 設定
    /////////////


    // const setServerUrl = useMemo(() => {
    //     return (url: string) => {
    //         if (!props.voiceChangerClient) return
    //         props.voiceChangerClient.setServerUrl(url, true)
    //         settingRef.current.mmvcServerUrl = url
    //         setSetting({ ...settingRef.current })
    //     }
    // }, [props.voiceChangerClient])

    const setSetting = useMemo(() => {
        return (setting: AudioStreamerSetting) => {
            if (!props.voiceChangerClient) return
            _setAudioStreamerSetting(setting)
            setItem(INDEXEDDB_KEY_CLIENT, setting)
            props.voiceChangerClient.updateAudioStreamerSetting(setting)
        }
    }, [props.voiceChangerClient])


    console.log("AUDIO STREAMER SETTING", audioStreamerSetting)
    return {
        audioStreamerSetting,
        clearSetting,
        setSetting,

    }
}