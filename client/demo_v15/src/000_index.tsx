import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { ErrorInfo, useMemo, useState, } from "react";
import { useMicrophoneOptions } from "./100_options_microphone";
import { AppStateProvider, useAppState } from "./001_provider/001_AppStateProvider";

import { library } from "@fortawesome/fontawesome-svg-core";
import { fas } from "@fortawesome/free-solid-svg-icons";
import { far } from "@fortawesome/free-regular-svg-icons";
import { fab } from "@fortawesome/free-brands-svg-icons";
import { AppRootProvider } from "./001_provider/001_AppRootProvider";
import ErrorBoundary from "./001_provider/900_ErrorBoundary";
import { INDEXEDDB_KEY_CLIENT, INDEXEDDB_KEY_MODEL_DATA, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_WORKLET, INDEXEDDB_KEY_WORKLETNODE, useIndexedDB } from "@dannadori/voice-changer-client-js";
import { CLIENT_TYPE, INDEXEDDB_KEY_AUDIO_OUTPUT, isDesktopApp } from "./const";
import { Dialog } from "./components/201_Dialog";

library.add(fas, far, fab);


const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {
    const appState = useAppState()
    const { removeItem } = useIndexedDB({ clientType: CLIENT_TYPE })
    const { voiceChangerSetting } = useMicrophoneOptions()
    const titleRow = useMemo(() => {
        const githubLink = isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://github.com/w-okada/voice-changer") }}>
                    <img src="./assets/icons/github.svg" />
                    <div className="tooltip-text">github</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                    <img src="./assets/icons/github.svg" />
                    <div className="tooltip-text">github</div>
                </a>
            )

        const manualLink = isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://zenn.dev/wok/books/0003_vc-helper-v_1_5") }}>
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text">manual</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://zenn.dev/wok/books/0003_vc-helper-v_1_5" target="_blank" rel="noopener noreferrer">
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text">manual</div>
                </a>
            )

        const toolLink = isDesktopApp() ?
            (
                <div className="link tooltip">
                    <img src="./assets/icons/tool.svg" />
                    <div className="tooltip-text tooltip-text-100px">
                        <p onClick={() => {
                            // @ts-ignore
                            window.electronAPI.openBrowser("https://w-okada.github.io/screen-recorder-ts/")
                        }}>
                            screen capture
                        </p>
                    </div>
                </div>
            )
            :
            (
                <div className="link tooltip">
                    <img src="./assets/icons/tool.svg" />
                    <div className="tooltip-text tooltip-text-100px">
                        <p onClick={() => {
                            window.open("https://w-okada.github.io/screen-recorder-ts/", '_blank', "noreferrer")
                        }}>
                            screen capture
                        </p>
                    </div>
                </div>
            )

        const coffeeLink = isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://www.buymeacoffee.com/wokad") }}>
                    <img className="donate-img" src="./assets/buymeacoffee.png" />
                    <div className="tooltip-text tooltip-text-100px">donate(寄付)</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://www.buymeacoffee.com/wokad" target="_blank" rel="noopener noreferrer">
                    <img className="donate-img" src="./assets/buymeacoffee.png" />
                    <div className="tooltip-text tooltip-text-100px">
                        donate(寄付)
                    </div>
                </a>
            )

        const licenseButton = (
            <span className="link" onClick={() => {
                document.getElementById("dialog")?.classList.add("dialog-container-show")
                appState.frontendManagerState.stateControls.showLicenseCheckbox.updateState(true)
            }}>
                <span>License</span>
            </span>
        )

        return (
            <div className="top-title">
                <span className="title">Voice Changer Setting</span>
                <span className="top-title-version">for MMVC v.1.5.x</span>
                <span className="belongings">
                    {githubLink}
                    {manualLink}
                    {toolLink}
                    {coffeeLink}
                    {licenseButton}
                </span>
                <span className="belongings">

                </span>
            </div>
        )
    }, [])

    const clearRow = useMemo(() => {
        const onClearSettingClicked = async () => {
            await appState.clearSetting()
            await removeItem(INDEXEDDB_KEY_AUDIO_OUTPUT)
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
                    <Dialog />
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
    // エラーバウンダリー設定
    const [error, setError] = useState<{ error: Error, errorInfo: ErrorInfo }>()
    const { removeItem } = useIndexedDB({ clientType: CLIENT_TYPE })

    const errorComponent = useMemo(() => {
        const errorName = error?.error.name || "no error name"
        const errorMessage = error?.error.message || "no error message"
        const errorInfos = (error?.errorInfo.componentStack || "no error stack").split("\n")

        const onClearCacheClicked = async () => {
            const indexedDBKeys = [
                INDEXEDDB_KEY_CLIENT,
                INDEXEDDB_KEY_SERVER,
                INDEXEDDB_KEY_WORKLETNODE,
                INDEXEDDB_KEY_MODEL_DATA,
                INDEXEDDB_KEY_WORKLET,
                INDEXEDDB_KEY_AUDIO_OUTPUT
            ]
            for (const k of indexedDBKeys) {
                await removeItem(k)
            }

            location.reload();
        }
        return (
            <div className="error-container">
                <div className="top-error-message">
                    ちょっと問題が起きたみたいです。
                </div>
                <div className="top-error-description">
                    <p>このアプリで管理している情報をクリアすると回復する場合があります。</p>
                    <p>下記のボタンを押して情報をクリアします。</p>
                    <p><button onClick={onClearCacheClicked}>アプリを初期化</button></p>
                </div>
                <div className="error-detail">
                    <div className="error-name">
                        {errorName}
                    </div>
                    <div className="error-message">
                        {errorMessage}
                    </div>
                    <div className="error-info-container">
                        {errorInfos.map(x => {
                            return <div className="error-info-line" key={x}>{x}</div>
                        })}
                    </div>

                </div>
            </div>
        )
    }, [error])

    const updateError = (error: Error, errorInfo: React.ErrorInfo) => {
        console.log("error compo", error, errorInfo)
        setError({ error, errorInfo })
    }

    return (
        <ErrorBoundary fallback={errorComponent} onError={updateError}>
            <AppStateProvider>
                <App></App>
            </AppStateProvider>
        </ErrorBoundary>
    )
}

root.render(
    <AppRootProvider>
        <AppStateWrapper></AppStateWrapper>
    </AppRootProvider>
);

