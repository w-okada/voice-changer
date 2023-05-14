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
        const onDownloadClicked = () => {
            const slot = appState.serverSetting.serverSetting.modelSlotIndex
            const model = appState.serverSetting.serverSetting.modelSlots[slot]

            const a = document.createElement("a")
            a.href = model.modelFile
            a.download = a.href.replace(/^.*[\\\/]/, '');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            guiState.stateControls.showWaitingCheckbox.updateState(false)
        }

        const exportOnnx = appState.serverSetting.serverSetting.framework == "PyTorch" ? (
            <div className="body-button left-margin-1" onClick={onnxExportButtonAction}>export onnx</div>
        ) : <></>

        return (
            <>
                <div className="body-row split-3-7 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">Export ONNX</div>
                    <div className="body-button-container">
                        {exportOnnx}
                        <div className="body-button left-margin-1" onClick={() => {
                            onDownloadClicked()
                        }}>download</div>
                    </div>
                </div>
            </>
        )
    }, [appState.getInfo, appState.serverSetting.serverSetting])

    return onnxExporthRow
}

