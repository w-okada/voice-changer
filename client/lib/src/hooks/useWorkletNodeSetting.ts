import { useState, useMemo } from "react"

import { WorkletNodeSetting } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"


export type UseWorkletNodeSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    defaultWorkletNodeSetting: WorkletNodeSetting
}

export type WorkletNodeSettingState = {
    workletNodeSetting: WorkletNodeSetting;
    updateWorkletNodeSetting: (setting: WorkletNodeSetting) => void
    startOutputRecording: () => void
    stopOutputRecording: () => Promise<Float32Array>
    trancateBuffer: () => Promise<void>
}

export const useWorkletNodeSetting = (props: UseWorkletNodeSettingProps): WorkletNodeSettingState => {

    const [workletNodeSetting, _setWorkletNodeSetting] = useState<WorkletNodeSetting>(props.defaultWorkletNodeSetting)

    //////////////
    // 設定
    /////////////

    const updateWorkletNodeSetting = useMemo(() => {
        return (_workletNodeSetting: WorkletNodeSetting) => {
            if (!props.voiceChangerClient) return
            for (let k in _workletNodeSetting) {
                const cur_v = workletNodeSetting[k as keyof WorkletNodeSetting]
                const new_v = _workletNodeSetting[k as keyof WorkletNodeSetting]
                if (cur_v != new_v) {
                    _setWorkletNodeSetting(_workletNodeSetting)
                    props.voiceChangerClient.updateWorkletNodeSetting(_workletNodeSetting)
                    break
                }
            }
        }
    }, [props.voiceChangerClient, workletNodeSetting])

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
        workletNodeSetting,
        updateWorkletNodeSetting,
        startOutputRecording,
        stopOutputRecording,
        trancateBuffer
    }
}