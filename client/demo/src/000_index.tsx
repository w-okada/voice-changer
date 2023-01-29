import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { useMemo, } from "react";
import { useMicrophoneOptions } from "./100_options_microphone";

const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {

    const { voiceChangerSetting, clearSetting } = useMicrophoneOptions()

    const onClearSettingClicked = async () => {
        clearSetting()
        location.reload()
    }

    const clearRow = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-3-4 left-padding-1">
                    <div className="body-button-container">
                        <div className="body-button" onClick={onClearSettingClicked}>clear setting</div>
                    </div>
                    <div className="body-item-text"></div>
                    <div className="body-item-text"></div>
                </div>
            </>
        )
    }, [])


    return (
        <div className="main-body">
            <div className="body-row split-6-4">
                <div className="body-top-title">
                    Voice Changer Setting
                </div>
                <div className="body-top-title-belongings">
                    <div className="belonging-item">
                        <a className="link" href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                            <img src="./assets/icons/github.svg" />
                            <span>github</span>
                        </a>
                    </div>
                    <div className="belonging-item">
                        <a className="link" href="https://zenn.dev/wok/articles/s01_vc001_top" target="_blank" rel="noopener noreferrer">
                            <img src="./assets/icons/help-circle.svg" />
                            <span>manual</span>
                        </a>
                    </div>

                </div>
            </div>
            {clearRow}
            {voiceChangerSetting}
            <div>
                <audio id="audio-output"></audio>
            </div>
        </div>
    )
}

root.render(
    <App></App>
);
