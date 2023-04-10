import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { ErrorInfo, useEffect, useMemo, useState, } from "react";
import { AppStateProvider } from "./001_provider/001_AppStateProvider";

import { library } from "@fortawesome/fontawesome-svg-core";
import { fas } from "@fortawesome/free-solid-svg-icons";
import { far } from "@fortawesome/free-regular-svg-icons";
import { fab } from "@fortawesome/free-brands-svg-icons";
import { AppRootProvider, useAppRoot } from "./001_provider/001_AppRootProvider";
import ErrorBoundary from "./001_provider/900_ErrorBoundary";
import { ClientType, INDEXEDDB_KEY_CLIENT, INDEXEDDB_KEY_MODEL_DATA, INDEXEDDB_KEY_SERVER, INDEXEDDB_KEY_WORKLET, INDEXEDDB_KEY_WORKLETNODE, useIndexedDB } from "@dannadori/voice-changer-client-js";
import { INDEXEDDB_KEY_AUDIO_OUTPUT, INDEXEDDB_KEY_DEFAULT_MODEL_TYPE } from "./const";
import { Demo } from "./components/demo/010_Demo";
import { ClientSelector } from "./001_ClientSelector";

library.add(fas, far, fab);


const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {
    const { appGuiSettingState } = useAppRoot()
    const front = useMemo(() => {
        if (appGuiSettingState.appGuiSetting.type == "demo") {
            return <Demo></Demo>
        } else {
            return <>unknown gui type. {appGuiSettingState.appGuiSetting.type}</>
        }
    }, [appGuiSettingState.appGuiSetting.type])

    return (
        <>
            {front}
        </>
    )
}

const AppStateWrapper = () => {
    const { appGuiSettingState, clientType, setClientType } = useAppRoot()
    // エラーバウンダリー設定
    const [error, setError] = useState<{ error: Error, errorInfo: ErrorInfo }>()
    const { removeItem } = useIndexedDB({ clientType: clientType })
    const { getItem } = useIndexedDB({ clientType: null })
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


    useEffect(() => {
        const loadDefaultModelType = async () => {
            const defaultModelType = await getItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE)
            if (defaultModelType && defaultModelType != "null") {
                setClientType(defaultModelType as ClientType)
            }
        }
        loadDefaultModelType()
    }, [])

    if (!clientType) {
        return <ClientSelector></ClientSelector>

    } else if (!appGuiSettingState.guiSettingLoaded) {
        return <></>
    } else {
        return (
            <ErrorBoundary fallback={errorComponent} onError={updateError}>
                <AppStateProvider>
                    <App></App>
                </AppStateProvider>
            </ErrorBoundary>
        )
    }

}

root.render(
    <AppRootProvider>
        <AppStateWrapper></AppStateWrapper>
    </AppRootProvider>
);

