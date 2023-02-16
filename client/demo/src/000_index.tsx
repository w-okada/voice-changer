import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { useMemo, } from "react";
import { useMicrophoneOptions } from "./100_options_microphone";
import { AppStateProvider, useAppState } from "./001_provider/001_AppStateProvider";


import { library } from "@fortawesome/fontawesome-svg-core";
import { fas } from "@fortawesome/free-solid-svg-icons";
import { far } from "@fortawesome/free-regular-svg-icons";
import { fab } from "@fortawesome/free-brands-svg-icons";
import { AppRootProvider } from "./001_provider/001_AppRootProvider";

library.add(fas, far, fab);


const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {
    const appState = useAppState()
    const { voiceChangerSetting } = useMicrophoneOptions()

    const titleRow = useMemo(() => {
        return (
            <div className="top-title">
                <span className="title">Voice Changer Setting</span>
                <span className="top-title-version">for v.1.5.x</span>
                <span className="belongings">
                    <a className="link" href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                        <img src="./assets/icons/github.svg" />
                        <span>github</span>
                    </a>
                    <a className="link" href="https://zenn.dev/wok/books/0003_vc-helper-v_1_5" target="_blank" rel="noopener noreferrer">
                        <img src="./assets/icons/help-circle.svg" />
                        <span>manual</span>
                    </a>
                    <a className="link" href="https://www.buymeacoffee.com/wokad" target="_blank" rel="noopener noreferrer">
                        <img className="donate-img" src="./assets/buymeacoffee.png" />
                        <span>コーヒーを寄付</span>
                    </a>

                </span>
                <span className="belongings">

                </span>
            </div>
        )
    }, [])

    const clearRow = useMemo(() => {
        const onClearSettingClicked = async () => {
            await appState.clearSetting()
            location.reload()
        }
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

    const mainSetting = useMemo(() => {
        return (
            <>
                <div className="main-body">
                    {titleRow}
                    {clearRow}
                    {voiceChangerSetting}
                </div>
            </>

        )
    }, [voiceChangerSetting])
    return (
        <>
            {mainSetting}
        </>
    )
}

const AppStateWrapper = () => {
    // const appRoot = useAppRoot()
    // if (!appRoot.audioContextState.audioContext) {
    //     return <>please click window</>
    // }
    return (
        <AppStateProvider>
            <App></App>
        </AppStateProvider>
    )
}

root.render(
    <AppRootProvider>
        <AppStateWrapper></AppStateWrapper>
    </AppRootProvider>
);
