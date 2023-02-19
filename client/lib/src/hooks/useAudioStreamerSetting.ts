import { useState, useMemo, useEffect } from "react"

import { INDEXEDDB_KEY_STREAMER, AudioStreamerSetting, DefaultAudioStreamerSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"

export type UseAudioStreamerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type AudioStreamerSettingState = {
    audioStreamerSetting: AudioStreamerSetting;
    clearSetting: () => Promise<void>
    updateAudioStreamerSetting: (setting: AudioStreamerSetting) => void

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


    const updateAudioStreamerSetting = useMemo(() => {
        return (_audioStreamerSetting: AudioStreamerSetting) => {
            if (!props.voiceChangerClient) return
            for (let k in _audioStreamerSetting) {
                const cur_v = audioStreamerSetting[k]
                const new_v = _audioStreamerSetting[k]
                if (cur_v != new_v) {
                    _setAudioStreamerSetting(_audioStreamerSetting)
                    setItem(INDEXEDDB_KEY_STREAMER, _audioStreamerSetting)
                    props.voiceChangerClient.updateAudioStreamerSetting(_audioStreamerSetting)
                    break
                }
            }
        }
    }, [props.voiceChangerClient, audioStreamerSetting])


    return {
        audioStreamerSetting,
        clearSetting,
        updateAudioStreamerSetting,

    }
}