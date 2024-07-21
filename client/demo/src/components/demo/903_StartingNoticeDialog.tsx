import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { isDesktopApp } from "../../const";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";
import { useMessageBuilder } from "../../hooks/useMessageBuilder";

export const StartingNoticeDialog = () => {
    const guiState = useGuiState();
    const { appGuiSettingState } = useAppRoot();

    const messageBuilderState = useMessageBuilder();
    useMemo(() => {
        messageBuilderState.setMessage(__filename, "directml_1", { ja: "directML版は実験的バージョンです。以下の既知の問題があります。", en: "DirectML version is an experimental version. There are the known issues as follows." });
        messageBuilderState.setMessage(__filename, "directml_2", {
            ja: "(1) 一部の設定変更を行うとgpuを使用していても変換処理が遅くなることが発生します。もしこの現象が発生したらGPUの値を-1にしてから再度0に戻してください。",
            en: "(1) When some settings are changed, conversion process becomes slow even when using GPU. If this occurs, reset the GPU value to -1 and then back to 0.",
        });
    }, []);

    const githubLink = isDesktopApp() ? (
        // @ts-ignore
        <span
            className="link"
            onClick={() => {
                // @ts-ignore
                window.electronAPI.openBrowser("https://github.com/deiteris/voice-changer");
            }}
        >
            Click here
        </span>
    ) : (
        <a className="link" href="https://github.com/deiteris/voice-changer" target="_blank" rel="noopener noreferrer">
            Click here
        </a>
    )

    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text"></div>
                <div className="body-button-container body-button-container-space-around">
                    <div
                        className="body-button"
                        onClick={() => {
                            guiState.stateControls.showStartingNoticeCheckbox.updateState(false);
                        }}
                    >
                        Start
                    </div>
                </div>
                <div className="body-item-text"></div>
            </div>
        );

        const welcomeMessage = (
            <div className="dialog-content-part">
                <div>Thank you for using the application! If you like the application or encounter any issues, please leave a star or file an issue in the Github repository.</div>
                <div>{githubLink} to visit the Github repository.</div>
            </div>
        );

        const directMLMessage = (
            <div className="dialog-content-part">
                <div>{messageBuilderState.getMessage(__filename, "directml_1")}</div>
                <div className="left-padding-1">{messageBuilderState.getMessage(__filename, "directml_2")}</div>
            </div>
        );

        const clickToStartMessage = (
            <div className="dialog-content-part">
                <div>Click "Start" to start using the application.</div>
            </div>
        );

        const edition = appGuiSettingState.edition;
        const content = (
            <div className="body-row">
                {welcomeMessage}
                {edition.indexOf("DirectML") >= 0 ? directMLMessage : <></>}
                {clickToStartMessage}
            </div>
        );

        return (
            <div className="dialog-frame">
                <div className="dialog-title">Welcome to Realtime Voice Changer!</div>
                <div className="dialog-content">
                    {content}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, [appGuiSettingState.edition]);
    return dialog;
};
