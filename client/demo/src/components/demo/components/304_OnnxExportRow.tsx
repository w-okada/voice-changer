import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { OnnxExporterInfo } from "@dannadori/voice-changer-client-js";
import { useGuiState } from "../001_GuiStateProvider";
export type OnnxExportRowProps = {
}

export const OnnxExportRow = (_props: OnnxExportRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()

    const onnxExportRow = useMemo(() => {

        const onnxExportButtonClassName = "body-button"
        const onnxExportButtonAction = async () => {

            if (guiState.isConverting) {
                alert("cannot export onnx when voice conversion is enabled")
                return
            }
            document.getElementById("dialog")?.classList.add("dialog-container-show")
            guiState.stateControls.showWaitingCheckbox.updateState(true)
            const res = await appState.serverSetting.getOnnx() as OnnxExporterInfo
            const a = document.createElement("a")
            a.href = res.path
            a.download = res.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            guiState.stateControls.showWaitingCheckbox.updateState(false)

        }
        const onnxExportButtonLabel = "onnx export"


        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Onnx Exporter</div>
                <div className="body-item-text">
                    <div></div>
                </div>
                <div className="body-button-container">
                    <div className={onnxExportButtonClassName} onClick={onnxExportButtonAction}>{onnxExportButtonLabel}</div>

                </div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, guiState.isConverting])

    return onnxExportRow
}