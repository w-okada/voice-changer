import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export const PyTorchClusterSelectRow = () => {
    const appState = useAppState()

    const pyTorchSelectRow = useMemo(() => {
        const clusterModelFilenameText = appState.serverSetting.fileUploadSetting.clusterTorchModel?.filename || appState.serverSetting.fileUploadSetting.clusterTorchModel?.file?.name || ""
        const onClusterFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pt") == false) {
                alert("モデルファイルの拡張子はptである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                clusterTorchModel: {
                    file: file
                }
            })
        }

        const onClusterFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting({
                ...appState.serverSetting.fileUploadSetting,
                clusterTorchModel: null
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">cluster(.pt)</div>
                <div className="body-item-text">
                    <div>{clusterModelFilenameText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onClusterFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onClusterFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [appState.serverSetting.fileUploadSetting, appState.serverSetting.setFileUploadSetting])

    return pyTorchSelectRow
}