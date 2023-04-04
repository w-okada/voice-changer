import React, { useMemo } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const ServerURLRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const advancedSetting = appGuiSettingState.appGuiSetting.front.advancedSetting

    const serverUrlRow = useMemo(() => {
        if (!advancedSetting.serverURLEnable) {
            return <></>
        }
        const onSetServerClicked = async () => {
            const input = document.getElementById("mmvc-server-url") as HTMLInputElement
            appState.clientSetting.setServerUrl(input.value)
        }
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">MMVC Server</div>
                <div className="body-input-container">
                    <input type="text" defaultValue={appState.workletNodeSetting.workletNodeSetting.serverUrl} id="mmvc-server-url" className="body-item-input" />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetServerClicked}>set</div>
                </div>
            </div>
        )
    }, [appState.workletNodeSetting.workletNodeSetting, appState.clientSetting.setServerUrl])

    return serverUrlRow
}