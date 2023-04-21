import React, { useMemo } from "react"
import { fileSelector } from "@dannadori/voice-changer-client-js"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export const PyTorchClusterSelectRow = () => {
    const appState = useAppState()
    const guiState = useGuiState()


    const pyTorchSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const clusterModelFilenameText = appState.serverSetting.fileUploadSettings[slot]?.clusterTorchModel?.filename || appState.serverSetting.fileUploadSettings[slot]?.clusterTorchModel?.file?.name || ""
        const onClusterFileLoadClicked = async () => {
            const file = await fileSelector("")
            if (file.name.endsWith(".pt") == false) {
                alert("モデルファイルの拡張子はptである必要があります。")
                return
            }
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
                clusterTorchModel: {
                    file: file
                }
            })
        }

        const onClusterFileClearClicked = () => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot],
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
    }, [appState.serverSetting.fileUploadSettings, appState.serverSetting.setFileUploadSetting, guiState.modelSlotNum])

    return pyTorchSelectRow
}