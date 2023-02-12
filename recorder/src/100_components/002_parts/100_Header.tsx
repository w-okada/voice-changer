import React, { useMemo } from "react";
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { RightSidebarButton } from "./102_RightSidebarButton";
import { DeviceSelector } from "./103_DeviceSelector";


export const Header = () => {
    const { applicationSetting } = useAppSetting()
    const header = useMemo(() => {
        return (
            <div className="header">
                <div className="header-application-title-container">
                    <img src="./assets/icons/zun.png" className="header-application-title-logo"></img>
                    <div className="header-application-title-text">Corpus Voice Recorder</div>
                </div>
                <div className="header-device-selector-container">
                    <div className="header-device-selector-text">Mic:</div>
                    <DeviceSelector deviceType={"audioinput"}></DeviceSelector>
                    <div className="header-device-selector-text">Sample Rate:{applicationSetting.applicationSetting.sample_rate}Hz</div>
                    <div className="header-device-selector-text">Sample Depth:16bit</div>
                    <div className="header-device-selector-text">Speaker:</div>
                    <DeviceSelector deviceType={"audiooutput"}></DeviceSelector>
                </div>
                <div className="header-button-container">
                    <a className="header-button-link" href="https://github.com/w-okada/voice-changer/wiki" target="_blank" rel="noopener noreferrer">
                        <img src="./assets/icons/help-circle.svg" />
                    </a>
                    <div className="header-button-spacer"></div>
                    <a className="header-button-link" href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                        <img src="./assets/icons/github.svg" />
                    </a>
                    <div className="header-button-spacer"></div>
                    <RightSidebarButton></RightSidebarButton>
                </div>
            </div>

        )
    }, [])

    return header;
};
