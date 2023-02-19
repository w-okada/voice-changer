import { useState, useMemo, useEffect } from "react"

import { DefaultWorkletNodeSetting, INDEXEDDB_KEY_WORKLETNODE, WorkletNodeSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"

export type UseWorkletNodeSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type WorkletNodeSettingState = {
    workletNodeSetting: WorkletNodeSetting;
    clearSetting: () => Promise<void>
    updateWorkletNodeSetting: (setting: WorkletNodeSetting) => void

}

export const useWorkletNodeSetting = (props: UseWorkletNodeSettingProps): WorkletNodeSettingState => {
    const [workletNodeSetting, _setWorkletNodeSetting] = useState<WorkletNodeSetting>(DefaultWorkletNodeSetting)
    const { setItem, getItem, removeItem } = useIndexedDB()

    // 初期化 その１ DBから取得
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_WORKLETNODE) as WorkletNodeSetting
            if (setting) {
                _setWorkletNodeSetting(setting)
            }
        }
        loadCache()
    }, [])

    // 初期化 その２ クライアントに設定
    useEffect(() => {
        if (!props.voiceChangerClient) return
        props.voiceChangerClient.setServerUrl(workletNodeSetting.serverUrl)
        props.voiceChangerClient.updateWorkletNodeSetting(workletNodeSetting)
    }, [props.voiceChangerClient])



    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_WORKLETNODE)
    }

    //////////////
    // 設定
    /////////////

    const updateWorkletNodeSetting = useMemo(() => {
        return (_workletNodeSetting: WorkletNodeSetting) => {
            if (!props.voiceChangerClient) return
            for (let k in _workletNodeSetting) {
                const cur_v = workletNodeSetting[k]
                const new_v = _workletNodeSetting[k]
                if (cur_v != new_v) {
                    _setWorkletNodeSetting(_workletNodeSetting)
                    setItem(INDEXEDDB_KEY_WORKLETNODE, _workletNodeSetting)
                    props.voiceChangerClient.updateWorkletNodeSetting(_workletNodeSetting)
                    break
                }
            }
        }
    }, [props.voiceChangerClient, workletNodeSetting])


    return {
        workletNodeSetting,
        clearSetting,
        updateWorkletNodeSetting,

    }
}