import React, { useMemo, useEffect } from "react"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"
import { useGuiState } from "./001_GuiStateProvider"

export const ModelUploaderRow = () => {
    const guiState = useGuiState()
    const { appGuiSettingState } = useAppRoot()
    useEffect(() => {
        if (appGuiSettingState.appGuiSetting.front.modelSetting.showPyTorchDefault) {
            guiState.setShowPyTorchModelUpload(true)
        }
    }, [])

    const modelUploaderRow = useMemo(() => {
        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Model Uploader</div>
                <div className="body-item-text">
                    <div></div>
                </div>
                <div className="body-item-text">
                    <div>
                        <input type="checkbox" checked={guiState.showPyTorchModelUpload} onChange={(e) => {
                            guiState.setShowPyTorchModelUpload(e.target.checked)
                        }} /> enable PyTorch
                    </div>
                </div>
            </div>
        )
    }, [guiState.showPyTorchModelUpload])

    return modelUploaderRow
}