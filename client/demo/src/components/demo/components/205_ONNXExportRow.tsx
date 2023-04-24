import { OnnxExporterInfo } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"


export type ONNXExportRowProps = {
}

export const ONNXExportRow = (_props: ONNXExportRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()

    const onnxExporthRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.framework != "PyTorch") {
            return <></>
        }

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
        return (
            <>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Export ONNX</div>
                    <div className="body-button-container">
                        <div className="body-button left-margin-1" onClick={onnxExportButtonAction}>export onnx</div>
                    </div>
                </div>
            </>
        )
    }, [appState.getInfo, appState.serverSetting.serverSetting])

    return onnxExporthRow
}

