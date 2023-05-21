import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type InputChunkNumRowProps = {
    nums: number[]
}
export const InputChunkNumRow = (props: InputChunkNumRowProps) => {
    const appState = useAppState()

    const inputChunkNumRow = useMemo(() => {
        let nums: number[]
        if (!props.nums) {
            nums = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 2048]
        } else {
            nums = props.nums
        }
        return (
            <div className="body-row split-3-2-2-3 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Input Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <select className="body-select" value={appState.workletNodeSetting.workletNodeSetting.inputChunkNum} onChange={(e) => {
                        appState.workletNodeSetting.updateWorkletNodeSetting({ ...appState.workletNodeSetting.workletNodeSetting, inputChunkNum: Number(e.target.value) })
                        appState.workletNodeSetting.trancateBuffer()
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverReadChunkSize: Number(e.target.value) })
                    }}>
                        {
                            nums.map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                    <div>buff: {(appState.workletNodeSetting.workletNodeSetting.inputChunkNum * 128 * 1000 / 48000).toFixed(1)}ms</div>
                </div>
                <div className="body-item-text">
                    <div>sample: {(appState.workletNodeSetting.workletNodeSetting.inputChunkNum * 128)}</div>
                </div>

            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting, appState.workletNodeSetting.updateWorkletNodeSetting, appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return inputChunkNumRow
}