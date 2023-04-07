import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type GPURowProps = {
}
export const GPURow = (_props: GPURowProps) => {
    const appState = useAppState()
    const gpuRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">GPU</div>
                <div className="body-input-container">
                    <input type="number" min={-2} max={5} step={1} value={appState.serverSetting.serverSetting.gpu} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, gpu: Number(e.target.value) })
                    }} />
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])
    return gpuRow
}