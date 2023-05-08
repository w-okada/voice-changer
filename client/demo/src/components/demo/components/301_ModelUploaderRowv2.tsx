import React, { useMemo } from "react"
import { useGuiState } from "../001_GuiStateProvider"

export type ModelUploaderRowv2Props = {}

export const ModelUploaderRowv2 = (_props: ModelUploaderRowv2Props) => {
    const guiState = useGuiState()

    const modelUploaderRow = useMemo(() => {

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Model Uploader</div>
                <div className="body-item-text">
                    <div></div>
                </div>
                <div className="body-item-text">
                </div>
            </div>
        )
    }, [guiState.showPyTorchModelUpload])

    return modelUploaderRow
}
