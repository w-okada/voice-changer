import React, { useMemo } from "react";
import { useGuiState } from "./001_GuiStateProvider";
import { getMessage } from "./messages/MessageBuilder";
import { isDesktopApp } from "../../const";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";


export const StartingNoticeDialog = () => {
    const guiState = useGuiState()
    const { appGuiSettingState } = useAppRoot()

    const coffeeLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://www.buymeacoffee.com/wokad") }}>
                    <img className="donate-img" src="./assets/buymeacoffee.png" /> donate
                    <div className="tooltip-text tooltip-text-100px">donate(寄付)</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://www.buymeacoffee.com/wokad" target="_blank" rel="noopener noreferrer">
                    <img className="donate-img" src="./assets/buymeacoffee.png" /> Donate
                    <div className="tooltip-text tooltip-text-100px">
                        donate(寄付)
                    </div>
                </a>
            )
    }, [])




    const dialog = useMemo(() => {
        const closeButtonRow = (
            <div className="body-row split-3-4-3 left-padding-1">
                <div className="body-item-text">
                </div>
                <div className="body-button-container body-button-container-space-around">
                    <div className="body-button" onClick={() => { guiState.stateControls.showStartingNoticeCheckbox.updateState(false) }} >start</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )

        const donationMessage = (
            <div className="dialog-content-part">
                <div>
                    {getMessage("donate_1")}
                </div>
                <div>
                    {coffeeLink}
                </div>
            </div>
        )

        const directMLMessage = (
            <div className="dialog-content-part">
                <div>
                    {getMessage("notice_1")}
                </div>
                <div className="left-padding-1">
                    {getMessage("notice_2")}
                </div>
            </div>
        )
        const clickToStartMessage = (
            <div className="dialog-content-part">
                <div>
                    {getMessage("click_to_start_1")}
                </div>
            </div>
        )
        const lang = window.navigator.language
        const edition = appGuiSettingState.edition
        const content = (
            <div className="body-row">
                {lang != "ja" || edition.indexOf("onnxdirectML-cuda") >= 0 ? donationMessage : <></>}
                {lang != "ja" || edition.indexOf("onnxdirectML-cuda") >= 0 ? directMLMessage : <></>}
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
