import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type DiffEnablerRowProps = {
}

export const DiffEnablerRow = (_props: DiffEnablerRowProps) => {
    const appState = useAppState()

    const diffEnablerRow = useMemo(() => {


        return (
            <div className="body-row split-3-2-2-2-1 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">DDSP Setting</div>
                <div>
                    <input type="checkbox" checked={appState.serverSetting.serverSetting.useEnhancer == 1} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, useEnhancer: e.target.checked ? 1 : 0 })
                    }} /> Enhancer
                </div>
                <div>
                    <input type="checkbox" checked={appState.serverSetting.serverSetting.useDiff == 1} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, useDiff: e.target.checked ? 1 : 0 })
                    }} /> Diff
                </div>
                <div>
                    <input type="checkbox" checked={appState.serverSetting.serverSetting.useDiffDpm == 1} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, useDiffDpm: e.target.checked ? 1 : 0 })
                    }} /> DiffDpm
                </div>
                <div>
                    <input type="checkbox" checked={appState.serverSetting.serverSetting.useDiffSilence == 1} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, useDiffSilence: e.target.checked ? 1 : 0 })
                    }} /> Silence
                </div>
            </div>
        )
    }, [
        appState.serverSetting.serverSetting,
        appState.serverSetting.updateServerSettings
    ])

    return diffEnablerRow
}


