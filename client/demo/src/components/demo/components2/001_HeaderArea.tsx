import React, { useMemo } from "react";
import { INDEXEDDB_KEY_AUDIO_OUTPUT, isDesktopApp } from "../../../const";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useIndexedDB } from "@dannadori/voice-changer-client-js";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";

export type HeaderAreaProps = {
    mainTitle: string;
    subTitle: string;
};

export const HeaderArea = (props: HeaderAreaProps) => {
    const { appGuiSettingState } = useAppRoot();
    const messageBuilderState = useMessageBuilder();
    const { clearSetting } = useAppState();

    const { removeItem, removeDB } = useIndexedDB({ clientType: null });

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "manual", { ja: "マニュアル", en: "manual" });
        messageBuilderState.setMessage(__filename, "support", { ja: "支援", en: "Donation" });
    }, []);

    const githubLink = isDesktopApp() ? (
        <span
            className="link tooltip"
            onClick={() => {
                // @ts-ignore
                window.electronAPI.openBrowser("https://github.com/deiteris/voice-changer");
            }}
        >
            <img src="./assets/icons/github.svg" />
            <div className="tooltip-text">Github</div>
        </span>
    ) : (
        <a className="link tooltip" href="https://github.com/deiteris/voice-changer" target="_blank" rel="noopener noreferrer">
            <img src="./assets/icons/github.svg" />
            <div className="tooltip-text">Github</div>
        </a>
    );

    // const manualLink = useMemo(() => {
    //     return isDesktopApp() ? (
    //         <span
    //             className="link tooltip"
    //             onClick={() => {
    //                 // @ts-ignore
    //                 window.electronAPI.openBrowser("https://github.com/deiteris/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md");
    //             }}
    //         >
    //             <img src="./assets/icons/help-circle.svg" />
    //             <div className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "manual")}</div>
    //         </span>
    //     ) : (
    //         <a className="link tooltip" href="https://github.com/deiteris/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md" target="_blank" rel="noopener noreferrer">
    //             <img src="./assets/icons/help-circle.svg" />
    //             <div className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "manual")}</div>
    //         </a>
    //     );
    // }, []);

    const headerArea = useMemo(() => {
        const onClearSettingClicked = async () => {
            await clearSetting();
            await removeItem(INDEXEDDB_KEY_AUDIO_OUTPUT);
            await removeDB();
            location.reload();
        };

        return (
            <div className="headerArea">
                <div className="title1">
                    <span className="title">{props.mainTitle}</span>
                    <span className="title-version">{props.subTitle}</span>
                    <span className="title-version-number">{appGuiSettingState.version}</span>
                    <span className="title-version-number">{appGuiSettingState.edition}</span>
                </div>
                <div className="icons">
                    <span className="belongings">
                        {githubLink}
                        {/* manualLink */}
                        {/* {licenseButton} */}
                    </span>
                    <span className="belongings">
                        <div className="belongings-button" onClick={onClearSettingClicked}>
                            clear setting
                        </div>
                        {/* <div className="belongings-button" onClick={onReloadClicked}>reload</div>
                        <div className="belongings-button" onClick={onReselectVCClicked}>select vc</div> */}
                    </span>
                </div>
            </div>
        );
    }, [props.subTitle, props.mainTitle, appGuiSettingState.version, appGuiSettingState.edition]);

    return headerArea;
};
