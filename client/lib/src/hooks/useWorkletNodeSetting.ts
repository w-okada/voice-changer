import { useState, useMemo, useEffect } from "react"

import { WorkletNodeSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"


export type UseWorkletNodeSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    defaultWorkletNodeSetting: WorkletNodeSetting
}

export type WorkletNodeSettingState = {
    startOutputRecording: () => void
    stopOutputRecording: () => Promise<Float32Array>
    trancateBuffer: () => Promise<void>
}

export const useWorkletNodeSetting = (props: UseWorkletNodeSettingProps): WorkletNodeSettingState => {
    // 更新比較用
    const [workletNodeSetting, _setWorkletNodeSetting] = useState<WorkletNodeSetting>(props.defaultWorkletNodeSetting)

    //////////////
    // 設定
    /////////////
    useEffect(() => {

        if (!props.voiceChangerClient) return
        for (let k in props.defaultWorkletNodeSetting) {
            const cur_v = workletNodeSetting[k as keyof WorkletNodeSetting]
            const new_v = props.defaultWorkletNodeSetting[k as keyof WorkletNodeSetting]
            if (cur_v != new_v) {
                _setWorkletNodeSetting(props.defaultWorkletNodeSetting)
                props.voiceChangerClient.updateWorkletNodeSetting(props.defaultWorkletNodeSetting)
                break
            }
        }

    }, [props.voiceChangerClient, props.defaultWorkletNodeSetting])


    const startOutputRecording = useMemo(() => {
        return () => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.startOutputRecording()
        }
    }, [props.voiceChangerClient])

    const stopOutputRecording = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return new Float32Array()
            return props.voiceChangerClient.stopOutputRecording()
        }
    }, [props.voiceChangerClient])

    const trancateBuffer = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.trancateBuffer()
        }
    }, [props.voiceChangerClient])

    return {
        startOutputRecording,
        stopOutputRecording,
        trancateBuffer
    }
}