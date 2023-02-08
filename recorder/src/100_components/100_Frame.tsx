import React, { useEffect } from "react";
import { AudioOutputElementId } from "../const";
import { useAppState } from "../003_provider/AppStateProvider";
import { useAppSetting } from "../003_provider/AppSettingProvider";
import { Header } from "./002_parts/100_Header";
import { Body } from "./002_parts/200_Body";
import { RightSidebar } from "./002_parts/300_RightSidebar";



export const Frame = () => {
    const { applicationSetting } = useAppSetting()
    const { frontendManagerState, corpusDataState, audioControllerState } = useAppState();


    // ハンドルキーボードイベント
    useEffect(() => {
        const keyDownHandler = (ev: KeyboardEvent) => {
            console.log("EVENT:", ev);
            const audioActive = audioControllerState.audioControllerState === "play" || audioControllerState.audioControllerState === "record";
            const unsavedRecord = audioControllerState.unsavedRecord;

            const key = ev.code;
            switch (key) {
                case "ArrowUp":
                case "ArrowLeft":
                    if (applicationSetting.applicationSetting.current_text_index > 0 && !audioActive && !unsavedRecord) {
                        applicationSetting.setCurrentTextIndex(applicationSetting.applicationSetting.current_text_index - 1);
                    }
                    return;
                case "ArrowDown":
                case "ArrowRight":
                    if (!applicationSetting.applicationSetting.current_text) {
                        return;
                    }
                    if (applicationSetting.applicationSetting.current_text_index < corpusDataState.corpusTextData[applicationSetting.applicationSetting.current_text].text.length - 1 && !audioActive && !unsavedRecord) {
                        applicationSetting.setCurrentTextIndex(applicationSetting.applicationSetting.current_text_index + 1);
                    }
                    return;
            }

            if (key === "Space") {
                //   let color = Math.floor(Math.random() * 0xFFFFFF).toString(16);
                //   for(let count = color.length; count < 6; count++) {
                //     color = '0' + color;
                //   }
                //   setBackgroundColor('#' + color);
            }
        };
        document.addEventListener("keydown", keyDownHandler, false);

        return () => {
            document.removeEventListener("keydown", keyDownHandler);
        };
    }, [applicationSetting.applicationSetting.current_text_index, audioControllerState.unsavedRecord, audioControllerState.audioControllerState]);


    return (
        <div>
            <audio src="" id={AudioOutputElementId}></audio>
            <div className="header-container">
                <Header></Header>
            </div>
            {frontendManagerState.stateControls.openRightSidebarCheckbox.trigger}
            <div className="body-container">
                <Body></Body>
            </div>
            {frontendManagerState.stateControls.openRightSidebarCheckbox.trigger}
            <div className="right-sidebar-container">
                <RightSidebar></RightSidebar>
            </div>
        </div>
    );
};
