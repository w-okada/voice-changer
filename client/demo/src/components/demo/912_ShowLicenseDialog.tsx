////// Currently not used /////////

import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { isDesktopApp } from "../../const";
import { useMessageBuilder } from "../../hooks/useMessageBuilder";

export const ShowLicenseDialog = () => {
    const guiState = useGuiState();

    const messageBuilderState = useMessageBuilder();
    useMemo(() => {
        messageBuilderState.setMessage(__filename, "nsf_hifigan1", {
            ja: "Diffusion SVC, DDSP SVCはvocodeerはDiffSinger Community Vocodersを使用しています。次のリンクからライセンスをご確認ください。",
            en: "Diffusion SVC and DDSP SVC uses DiffSinger Community Vocoders. Please check the license from the following link.",
        });
        messageBuilderState.setMessage(__filename, "nsf_hifigan2", { ja: "別のモデルを使用する場合はpretrain\\nsf_hifiganに設置してください。", en: "Please place it on pretrain\\nsf_hifigan if you are using a different model." });
    }, []);

    const hifiGanLink = useMemo(() => {
        return isDesktopApp() ? (
            // @ts-ignore
            <span
                className="link"
                onClick={() => {
                    // @ts-ignore
                    window.electronAPI.openBrowser("https://openvpi.github.io/vocoders/");
                }}
            >
                license
            </span>
        ) : (
            <a className="link" href="https://openvpi.github.io/vocoders/" target="_blank" rel="noopener noreferrer">
                license
            </a>
        );
    }, []);

    const dialog = useMemo(() => {
        const hifiganMessage = (
            <div className="dialog-content-part">
                <div>{messageBuilderState.getMessage(__filename, "nsf_hifigan1")}</div>
                <div>{messageBuilderState.getMessage(__filename, "nsf_hifigan2")}</div>
                <div>{hifiGanLink}</div>
            </div>
        );

        const buttonsRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text"></div>
                <div className="body-button-container body-button-container-space-around">
                    <div
                        className="body-button"
                        onClick={() => {
                            guiState.stateControls.showLicenseCheckbox.updateState(false);
                        }}
                    >
                        close
                    </div>
                </div>
                <div className="body-item-text"></div>
            </div>
        );

        return (
            <div className="dialog-frame">
                <div className="dialog-title">Input Dialog</div>
                <div className="dialog-content">
                    <div className="body-row">{hifiganMessage}</div>
                    {buttonsRow}
                </div>
            </div>
        );
    }, [guiState.textInputResolve]);
    return dialog;
};
