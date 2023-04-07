import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export type ServerInfoRowProps = {
}

export const ServerInfoRow = (_props: ServerInfoRowProps) => {
    const appState = useAppState()

    const serverInfoRow = useMemo(() => {
        const onReloadClicked = async () => {
            const info = await appState.getInfo()
            console.log("info", info)
        }
        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Model Info:</div>
                    <div className="body-item-text">
                        <span className="body-item-text-item">{appState.serverSetting.serverSetting.configFile || ""}</span>
                        <span className="body-item-text-item">{appState.serverSetting.serverSetting.pyTorchModelFile || ""}</span>
                        <span className="body-item-text-item">{appState.serverSetting.serverSetting.onnxModelFile || ""}</span>


                    </div>
                    <div className="body-button-container">
                        <div className="body-button" onClick={onReloadClicked}>reload</div>
                    </div>
                </div>
            </>
        )
    }, [appState.getInfo, appState.serverSetting.serverSetting])

    return serverInfoRow
}