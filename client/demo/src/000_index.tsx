import * as React from "react";
import { createRoot } from "react-dom/client";
import "./css/App.css"
import { useEffect, useMemo, useRef, useState } from "react";
import { useMicrophoneOptions } from "./100_options_microphone";
import { VoiceChnagerClient, createDummyMediaStream } from "@dannadori/voice-changer-client-js"
import { AUDIO_ELEMENT_FOR_PLAY_RESULT } from "./const";

const container = document.getElementById("app")!;
const root = createRoot(container);

const App = () => {

    const { voiceChangerSetting } = useMicrophoneOptions()

    const onClearSettingClicked = async () => {
        //@ts-ignore
        await chrome.storage.local.clear();
        //@ts-ignore
        await chrome.storage.sync.clear();

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
        <div className="body">
            <div className="body-row">
                <div className="body-top-title">
                    Voice Changer Setting
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
