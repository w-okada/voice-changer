import * as React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { AppStateProvider } from "./003_provider/AppStateProvider";
import { AppSettingProvider, useAppSetting } from "./003_provider/AppSettingProvider";

import "./100_components/001_css/001_App.css";


const AppStateProviderWrapper = () => {
    const { applicationSetting, deviceManagerState } = useAppSetting();
    const [firstTach, setFirstTouch] = React.useState<boolean>(false);

    if (!applicationSetting || !firstTach) {
        const clearSetting = () => {
            const result = window.confirm('Initialize settings.');
            if (result) {
                applicationSetting.clearSetting()
                location.reload()
            }
        }
        return (
            <div className="front-container">
                <div className="front-title">Corpus Voice Recorder</div>
                <div className="front-description">
                    <p>This app is a recording app for voice synthesis.</p>
                    <p>Runs entirely on the client. Data will not be uploaded to the server. Data is stored within your browser.</p>
                    <p>Click <a href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">here</a> for the source code and usage instructions.</p>
                    <p className="front-description-strong">If you've tried using it and think it would be nice to treat yourself to a cup of coffee, please support us here.</p>
                    <p>
                        <a href="https://www.buymeacoffee.com/wokad">
                            <img className="front-description-img" src="./coffee.png"></img>
                        </a>
                    </p>
                    <a></a>
                </div>
                <div
                    className="front-start-button front-start-button-color"
                    onClick={() => {
                        setFirstTouch(true);
                    }}
                >
                    Click to start
                </div>
                <div className="front-note">Confirmed operating environment: Windows 11 + Chrome</div>
                <div className="front-description">
                    <p>The emotion and recitation scripts of the ITA corpus are currently registered.</p>
                    <p>Since it is intended for use with <a href="https://github.com/isletennos/MMVC_Trainer" target="_blank">MMVC</a>, the recording settings are 48000Hz, 16bit.</p>
                    <p>(Converts to 24000Hz internally when exporting.)</p>

                </div>


                {/* <div className="front-attention">
                    <p>動作確認のため、少量の利用から始めて、こまめなExportをお願いします。</p>
                    <p>ブラウザでデータ削除を行うとデータ消えるので注意してください。</p>
                </div> */}
                <div className="front-disclaimer">Disclaimer: We are not responsible for any direct, indirect, ripple, consequential, or special damages arising from the use or inability to use this software</div>


                <div className="front-clear-setting" onClick={clearSetting}>
                    Clear Setting
                </div>
            </div>
        );
    } else if (deviceManagerState.audioInputDevices.length === 0) {
        return (
            <>
                <div className="start-button">Loading Devices...</div>
            </>
        );
    } else {
        return (
            <AppStateProvider>
                <App />
            </AppStateProvider>
        );
    }
};

const container = document.getElementById("app")!;
const root = createRoot(container);
root.render(
    <AppSettingProvider>
        <AppStateProviderWrapper></AppStateProviderWrapper>
    </AppSettingProvider>
);
