import React, { useMemo } from "react";
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { useAppState } from "../../003_provider/AppStateProvider";

const enabledButtonClass = "button";
const disabledButtonClass = "button disable";
const activeButtonClass = "button active";
const attentionButtonClass = "button attention";
type ButtonStates = {
    recordButtonClass: string;
    stopButtonClass: string;
    playButtonClass: string;
    keepButtonClass: string;
    dismissButtonClass: string;
    recordAction: () => void;
    stopAction: () => void;
    playAction: () => void;
    keepAction: () => void;
    dismissAction: () => void;
};

export const AudioController = () => {
    const { applicationSetting } = useAppSetting()
    const { audioControllerState, mediaRecorderState, waveSurferState } = useAppState();



    const { recordButton, stopButton, playButton, keepButton, dismissButton } = useMemo(() => {
        const buttonStates: ButtonStates = {
            recordButtonClass: "",
            stopButtonClass: "",
            playButtonClass: "",
            keepButtonClass: "",
            dismissButtonClass: "",

            recordAction: () => { },
            stopAction: () => { },
            playAction: () => { },
            keepAction: () => { },
            dismissAction: () => { },
        };
        switch (audioControllerState.audioControllerState) {
            case "stop":

                // ボタンの状態
                buttonStates.recordButtonClass = enabledButtonClass; // [action needed]
                buttonStates.stopButtonClass = activeButtonClass;
                if (audioControllerState.tempUserData.vfWavBlob) {
                    buttonStates.playButtonClass = enabledButtonClass; // [action needed]
                } else {
                    buttonStates.playButtonClass = disabledButtonClass;
                }
                if (audioControllerState.unsavedRecord) {
                    {
                        // セーブされていない新録がある場合。
                        buttonStates.keepButtonClass = attentionButtonClass; // [action needed]
                        buttonStates.dismissButtonClass = attentionButtonClass; // [action needed]
                    }
                } else {
                    buttonStates.keepButtonClass = disabledButtonClass;
                    buttonStates.dismissButtonClass = disabledButtonClass;
                }

                // ボタンのアクション
                buttonStates.recordAction = () => {
                    audioControllerState.setAudioControllerState("record");
                    mediaRecorderState.startRecord();
                };
                if (audioControllerState.tempUserData.vfWavBlob) {
                    // バッファ上に音声がある場合。（ローカルストレージ、新録両方。）
                    buttonStates.playAction = () => {
                        audioControllerState.setAudioControllerState("play");
                        waveSurferState.playRegion();
                    };
                }
                if (audioControllerState.unsavedRecord) {
                    // セーブされていない新録がある場合。
                    buttonStates.keepAction = () => {
                        audioControllerState.saveWavBlob()
                        audioControllerState.setUnsavedRecord(false);
                    };
                    buttonStates.dismissAction = () => {
                        audioControllerState.restoreFixedUserData()
                        audioControllerState.setUnsavedRecord(false);
                    };
                }
                break;
            case "record":
                buttonStates.recordButtonClass = activeButtonClass;
                buttonStates.stopButtonClass = enabledButtonClass; // [action needed]
                buttonStates.playButtonClass = disabledButtonClass;
                buttonStates.keepButtonClass = disabledButtonClass;
                buttonStates.dismissButtonClass = disabledButtonClass;
                buttonStates.stopAction = () => {
                    audioControllerState.setAudioControllerState("stop");
                    mediaRecorderState.pauseRecord();
                    const { micWavBlob, vfWavBlob, vfDuration, vfSamples, micSamples } = mediaRecorderState.getRecordedDataBlobs();
                    // const micSpec = drawMel(micSamples, applicationSetting.applicationSetting.sample_rate)
                    // const vfSpec = drawMel(vfSamples, applicationSetting.applicationSetting.sample_rate)
                    audioControllerState.setTempWavBlob(micWavBlob, vfWavBlob, micSamples, vfSamples, "", "", [0, vfDuration])

                    audioControllerState.setUnsavedRecord(true);
                    waveSurferState.loadMusic(vfWavBlob);
                };
                break;

            case "play":
                buttonStates.recordButtonClass = disabledButtonClass;
                buttonStates.stopButtonClass = enabledButtonClass; // [action needed]
                buttonStates.playButtonClass = activeButtonClass;
                buttonStates.keepButtonClass = disabledButtonClass;
                buttonStates.dismissButtonClass = disabledButtonClass;
                buttonStates.stopAction = () => {
                    waveSurferState.stop();
                    audioControllerState.setAudioControllerState("stop");
                };
                break;
        }
        const recordButton = (
            <div className={buttonStates.recordButtonClass} onClick={buttonStates.recordAction}>
                record
            </div>
        );
        const stopButton = (
            <div className={buttonStates.stopButtonClass} onClick={buttonStates.stopAction}>
                stop
            </div>
        );
        const playButton = (
            <div className={buttonStates.playButtonClass} onClick={buttonStates.playAction}>
                play
            </div>
        );
        const keepButton = (
            <div className={buttonStates.keepButtonClass} onClick={buttonStates.keepAction}>
                keep
            </div>
        );
        const dismissButton = (
            <div className={buttonStates.dismissButtonClass} onClick={buttonStates.dismissAction}>
                dismiss
            </div>
        );

        return { recordButton, stopButton, playButton, keepButton, dismissButton };
    }, [audioControllerState.audioControllerState, audioControllerState.tempUserData, applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index, mediaRecorderState.startRecord, mediaRecorderState.pauseRecord]);

    return (
        <>
            {recordButton} {stopButton} {playButton} {keepButton} {dismissButton}
        </>

    );
};
