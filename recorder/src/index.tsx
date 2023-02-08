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
            const result = window.confirm('設定を初期化します。');
            if (result) {
                applicationSetting.clearSetting()
                location.reload()
            }
        }
        return (
            <div className="front-container">
                <div className="front-title">Corpus Voice Recorder</div>
                <div className="front-description">
                    <p>このアプリは音声合成のためのレコーディングアプリです。</p>
                    <p>完全にクライアント上で動きます。サーバへのデータアップロードは行いません。データはブラウザ内に蓄積されます。</p>
                    <p>
                        ソースコード、使用方法は
                        <a href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">こちら。</a>
                    </p>
                    <p className="front-description-strong">使ってみてコーヒーくらいならごちそうしてもいいかなという人はこちらからご支援お願いします。 </p>
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
                <div className="front-note">確認動作環境:Windows 11 + Chrome</div>
                <div className="front-description">
                    <p>現在ITAコーパスのemotionとrecitationの台本が登録されています。</p>
                    <p>
                        <a href="https://github.com/isletennos/MMVC_Trainer" target="_blank">
                            MMVC
                        </a>
                        での使用を想定しているため、48000Hz, 16bitの録音設定になっています。
                    </p>
                    <p>
                        (エクスポート時に内部で24000Hzに変換します。)
                    </p>

                </div>


                {/* <div className="front-attention">
                    <p>動作確認のため、少量の利用から始めて、こまめなExportをお願いします。</p>
                    <p>ブラウザでデータ削除を行うとデータ消えるので注意してください。</p>
                </div> */}
                <div className="front-disclaimer">免責：本ソフトウェアの使用または使用不能により生じたいかなる直接損害・間接損害・波及的損害・結果的損害 または特別損害についても、一切責任を負いません。</div>


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
