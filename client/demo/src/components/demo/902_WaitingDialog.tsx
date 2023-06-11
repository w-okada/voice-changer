import React, { useMemo } from "react";
import { useMessageBuilder } from "../../hooks/useMessageBuilder";


export const WaitingDialog = () => {
    // const guiState = useGuiState()
    const messageBuilderState = useMessageBuilder()
    useMemo(() => {
        messageBuilderState.setMessage(__filename, "wait", { "ja": "しばらくお待ちください", "en": "please wait..." })
        messageBuilderState.setMessage(__filename, "wait_sub1", { "ja": "ONNXファイルを生成しています。", "en": "generating ONNX file." })
        messageBuilderState.setMessage(__filename, "wait_sub2", { "ja": "しばらくお待ちください(1分程度)。", "en": "please wait... (about 1 min)." })
    }, [])

    const dialog = useMemo(() => {
        // const closeButtonRow = (
        //     <div className="body-row split-3-4-3 left-padding-1">
        //         <div className="body-item-text">
        //         </div>
        //         <div className="body-button-container body-button-container-space-around">
        //             <div className="body-button" onClick={() => { guiState.stateControls.showWaitingCheckbox.updateState(false) }} >close</div>
        //         </div>
        //         <div className="body-item-text"></div>
        //     </div>
        // )
        const content = (
            <div className="body-row left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-item-text">
                    {messageBuilderState.getMessage(__filename, "wait_sub1")}
                </div>
                <div className="body-item-text">
                    {messageBuilderState.getMessage(__filename, "wait_sub2")}
                </div>
            </div>
        )

        return (
            <div className="dialog-frame">
                <div className="dialog-title">{messageBuilderState.getMessage(__filename, "wait")}</div>
                <div className="dialog-content">
                    {content}
                    {/* {closeButtonRow} */}
                </div>
            </div>
        );
    }, []);
    return dialog;

};
