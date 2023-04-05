import React, { useMemo } from "react"
import { Framework } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../001_provider/001_AppStateProvider"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

export const FrameworkRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const modelSetting = appGuiSettingState.appGuiSetting.front.modelSetting

    const frameworkRow = useMemo(() => {
        if (!modelSetting.frameworkEnable) {
            return <></>
        }
        const onFrameworkChanged = async (val: Framework) => {
            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, framework: val })
        }

        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Framework</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.framework} onChange={(e) => {
                        onFrameworkChanged(e.target.value as
                            Framework)
                    }}>
                        {
                            Object.values(Framework).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>

        )
    }, [appState.serverSetting.serverSetting.framework, appState.serverSetting.updateServerSettings])

    return frameworkRow
}