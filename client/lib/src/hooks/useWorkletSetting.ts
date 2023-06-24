import { useState, useMemo, useEffect } from "react"
import { WorkletSetting, DefaultWorkletSetting, INDEXEDDB_KEY_WORKLET } from "../const";
import { VoiceChangerClient } from "../VoiceChangerClient";
import { useIndexedDB } from "./useIndexedDB";

export type UseWorkletSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type WorkletSettingState = {
    setting: WorkletSetting;
    clearSetting: () => Promise<void>
    setSetting: (setting: WorkletSetting) => void;

}

export const useWorkletSetting = (props: UseWorkletSettingProps): WorkletSettingState => {
    const [setting, _setSetting] = useState<WorkletSetting>(DefaultWorkletSetting)
    const { setItem, getItem, removeItem } = useIndexedDB({ clientType: null })
    // DBから設定取得（キャッシュによる初期化）
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_WORKLET)
            if (!setting) {
                // デフォルト設定
                const params = new URLSearchParams(location.search);
                const colab = params.get("colab")
                if (colab == "true") {
                    _setSetting({
                        numTrancateTreshold: 300,
                        volTrancateThreshold: 0.0005,
                        volTrancateLength: 32,
                    })
                } else {
                    _setSetting({
                        numTrancateTreshold: 100,
                        volTrancateThreshold: 0.0005,
                        volTrancateLength: 32,
                    })
                }
            } else {
                _setSetting({
                    ...(setting as WorkletSetting)
                })
            }
        }
        loadCache()
    }, [])

    // クライアントへ設定反映  初期化, 設定変更
    useEffect(() => {
        if (!props.voiceChangerClient) return
        props.voiceChangerClient.configureWorklet(setting)
    }, [props.voiceChangerClient, setting])

    // 設定 _setSettingがトリガでuseEffectが呼ばれて、workletに設定が飛ぶ
    const setSetting = useMemo(() => {
        return (setting: WorkletSetting) => {
            if (!props.voiceChangerClient) return
            _setSetting(setting)
            setItem(INDEXEDDB_KEY_WORKLET, setting)
        }
    }, [props.voiceChangerClient])

    // その他 オペレーション
    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_WORKLET)
    }


    return {
        setting,
        clearSetting,
        setSetting,

    }
}