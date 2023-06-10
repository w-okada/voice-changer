import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";


export const TextInputDialog = () => {
    const guiState = useGuiState()

    const dialog = useMemo(() => {
        const buttonsRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => {
                        const inputText = document.getElementById("input-text") as HTMLInputElement
                        const text = inputText.value
                        inputText.value = ""
                        if (guiState.textInputResolve) {
                            guiState.textInputResolve.resolve!(text)
                            guiState.setTextInputResolve(null)
                        }
                        guiState.stateControls.showTextInputCheckbox.updateState(false)
                    }} >ok</div>
                    <div className="body-button" onClick={() => {
                        const inputText = document.getElementById("input-text") as HTMLInputElement
                        const text = inputText.value
                        inputText.value = ""
                        if (guiState.textInputResolve) {
                            guiState.textInputResolve.resolve!("")
                            guiState.setTextInputResolve(null)
                        }
                        guiState.stateControls.showTextInputCheckbox.updateState(false)
                    }} >cancel</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
        const textInput = (
            <div className="input-text-container">
                <div>Input Text: </div>
                <input id="input-text" type="text" />
            </div>
        )
        return (
            <div className="dialog-frame">
                <div className="dialog-title">Input Dialog</div>
                <div className="dialog-content">
                    {textInput}
                    {buttonsRow}
                </div>
            </div>
        );
    }, [guiState.textInputResolve]);
    return dialog;

};
