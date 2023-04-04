import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const ExtraDataLengthRow = () => {
    const appState = useAppState()
    const converterSetting = appState.appGuiSettingState.appGuiSetting.front.converterSetting


    const extraDataLengthRow = useMemo(() => {
        if (!converterSetting.extraDataLengthEnable) {
            return <></>
        }
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Extra Data Length</div>
                <div className="body-input-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.extraConvertSize} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, extraConvertSize: Number(e.target.value) })
                        appState.workletNodeSetting.trancateBuffer()
                    }}>
                        {
                            [1024 * 4, 1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128].map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return extraDataLengthRow
}