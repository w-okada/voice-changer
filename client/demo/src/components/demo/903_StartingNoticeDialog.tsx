import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
// import { getMessage } from "./messages/MessageBuilder";
import { isDesktopApp } from "../../const";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";
import { useMessageBuilder } from "../../hooks/useMessageBuilder";


export const StartingNoticeDialog = () => {
    const guiState = useGuiState()
    const { appGuiSettingState } = useAppRoot()

    const messageBuilderState = useMessageBuilder()
    useMemo(() => {
        messageBuilderState.setMessage(__filename, "support", { "ja": "支援", "en": "Donation" })
        messageBuilderState.setMessage(__filename, "support_message_1", { "ja": "このソフトウェアを気に入ったら開発者にコーヒーをご馳走してあげよう。黄色いアイコンから。", "en": "This software is supported by donations. Thank you for your support!" })
        messageBuilderState.setMessage(__filename, "support_message_2", { "ja": "コーヒーをご馳走する。", "en": "I will support a developer by buying coffee." })

        messageBuilderState.setMessage(__filename, "directml_1", { "ja": "directML版は実験的バージョンです。以下の既知の問題があります。", "en": "DirectML version is an experimental version. There are the known issues as follows." })
        messageBuilderState.setMessage(__filename, "directml_2", { "ja": "(1) 一部の設定変更を行うとgpuを使用していても変換処理が遅くなることが発生します。もしこの現象が発生したらGPUの値を-1にしてから再度0に戻してください。", "en": "(1) When some settings are changed, conversion process becomes slow even when using GPU. If this occurs, reset the GPU value to -1 and then back to 0." })
        messageBuilderState.setMessage(__filename, "click_to_start", { "ja": "スタートボタンを押してください。", "en": "Click to start" })
        messageBuilderState.setMessage(__filename, "start", { "ja": "スタート", "en": "start" })

    }, [])



    const coffeeLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link" onClick={() => { window.electronAPI.openBrowser("https://www.buymeacoffee.com/wokad") }}>
                    <img className="donate-img" src="./assets/buymeacoffee.png" /> {messageBuilderState.getMessage(__filename, "support_message_2")}
                </span>
            )
            :
            (
                <a className="link" href="https://www.buymeacoffee.com/wokad" target="_blank" rel="noopener noreferrer">
                    <img className="donate-img" src="./assets/buymeacoffee.png" /> {messageBuilderState.getMessage(__filename, "support_message_2")}
                </a>
            )
    }, [])




    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { guiState.stateControls.showStartingNoticeCheckbox.updateState(false) }} >
                        {messageBuilderState.getMessage(__filename, "start")}
                    </div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )

        const donationMessage = (
            <div className="dialog-content-part">
                <div>
                    {messageBuilderState.getMessage(__filename, "support_message_1")}
                </div>
                <div>
                    {coffeeLink}
                </div>
            </div>
        )

        const directMLMessage = (
            <div className="dialog-content-part">
                <div>
                    {messageBuilderState.getMessage(__filename, "directml_1")}
                </div>
                <div className="left-padding-1">
                    {messageBuilderState.getMessage(__filename, "directml_2")}
                </div>
            </div>
        )
        const clickToStartMessage = (
            <div className="dialog-content-part">
                <div>
                    {messageBuilderState.getMessage(__filename, "click_to_start")}
                </div>
            </div>
        )
        const edition = appGuiSettingState.edition
        const content = (
            <div className="body-row">
                {donationMessage}
                {edition.indexOf("onnxdirectML-cuda") >= 0 ? directMLMessage : <></>}
                {clickToStartMessage}
            </div>
        )

        return (
            <div className="dialog-frame">
                <div className="dialog-title">Message</div>
                <div className="dialog-content">
                    {content}
                    {closeButtonRow}
                </div>
            </div>
        );
    }, [appGuiSettingState.edition]);
    return dialog;

};
