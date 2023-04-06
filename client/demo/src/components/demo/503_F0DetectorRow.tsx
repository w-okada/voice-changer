import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"
import { F0Detector } from "@dannadori/voice-changer-client-js";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";

export const F0DetectorRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const qualityControlSetting = appGuiSettingState.appGuiSetting.front.qualityControl
    const f0DetectorRow = useMemo(() => {
        if (!qualityControlSetting.F0DetectorEnable) {
            return <></>
        }

        const desc = { "harvest": "High Quality", "dio": "Light Weight" }
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">F0 Detector</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.f0Detector} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, f0Detector: e.target.value as F0Detector })
                    }}>
                        {
                            Object.values(F0Detector).map(x => {
                                //@ts-ignore
                                return <option key={x} value={x}>{x}({desc[x]})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting.f0Detector, appState.serverSetting.updateServerSettings])


    return f0DetectorRow
}