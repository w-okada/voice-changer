import React, { useEffect, useMemo } from "react"
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { useAppState } from "../../003_provider/AppStateProvider";
import { generateEmptyWav } from "../../const";
import { MelSpectrogram } from "./207_MelSpectrogram";

export const WaveSurferView = () => {
    const { applicationSetting } = useAppSetting()
    const { audioControllerState, waveSurferState, } = useAppState();
    useEffect(() => {
        const loadMusic = async () => {
            if (!audioControllerState.tempUserData.vfWavBlob) {
                const dummy = generateEmptyWav()
                await waveSurferState.loadMusic(dummy);
                return
            }
            await waveSurferState.loadMusic(audioControllerState.tempUserData.vfWavBlob);
        }
        loadMusic()
    }, [audioControllerState.tempUserData]);

    useEffect(() => {
        const regionTimeDiv = document.getElementById("waveform-region-time") as HTMLDivElement;
        const timeDiv = document.getElementById("waveform-time") as HTMLDivElement;

        if (!audioControllerState.tempUserData.vfWavBlob) {
            waveSurferState.setRegion(0, 0);
            regionTimeDiv.innerText = `please record.`;
            timeDiv.innerText = `no record.`
            return
        }
        const start = audioControllerState.tempUserData.region ? audioControllerState.tempUserData.region[0] : 0;
        const end = audioControllerState.tempUserData.region ? audioControllerState.tempUserData.region[1] : 1;
        const dur = end - start;

        regionTimeDiv.innerText = `Region:${start.toFixed(2)} - ${end.toFixed(2)} [${dur.toFixed(2)}]`;
        waveSurferState.setRegion(start, end);

    }, [audioControllerState.tempUserData])

    // Wavesurfer
    useEffect(() => {
        const timeDiv = document.getElementById("waveform-time") as HTMLDivElement;
        waveSurferState.setListener({
            audioprocess: () => {
                const timeInfos = waveSurferState.getTimeInfos();
                if (timeInfos.totalTime < 1) {
                    timeDiv.className = "waveform-header-item-warn";
                    timeDiv.innerText = `WARNING!!! Under 1sec. Time:${timeInfos.currentTime.toFixed(2)} / ${timeInfos.totalTime.toFixed(2)}`;
                } else {
                    timeDiv.className = "waveform-header-item";
                    timeDiv.innerText = `Time:${timeInfos.currentTime.toFixed(2)} / ${timeInfos.totalTime.toFixed(2)}`;
                }
            },
            finish: () => {
                audioControllerState.setAudioControllerState("stop");
            },
            ready: () => {
                const timeInfos = waveSurferState.getTimeInfos();
                if (timeInfos.totalTime < 1) {
                    timeDiv.className = "waveform-header-item-warn";
                    timeDiv.innerText = `WARNING!!! Under 1sec. Time:${timeInfos.currentTime.toFixed(2)} / ${timeInfos.totalTime.toFixed(2)}`;
                } else {
                    timeDiv.className = "waveform-header-item";
                    timeDiv.innerText = `Time:${timeInfos.currentTime.toFixed(2)} / ${timeInfos.totalTime.toFixed(2)}`;
                }
            },
            regionUpdate: (start: number, end: number) => {
                if (!applicationSetting.applicationSetting.current_text) {
                    return;
                }
                audioControllerState.setTempWavBlob(audioControllerState.tempUserData.micWavBlob, audioControllerState.tempUserData.vfWavBlob, audioControllerState.tempUserData.micWavSamples, audioControllerState.tempUserData.vfWavSamples, audioControllerState.tempUserData.micSpec, audioControllerState.tempUserData.vfSpec, [start, end])
                audioControllerState.setUnsavedRecord(true);
            },
        });
    }, [waveSurferState.setListener, waveSurferState.getTimeInfos, applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index, audioControllerState.unsavedRecord, audioControllerState.tempUserData]);

    const view = useMemo(() => {
        return (
            <>
                <div className="height-40 waveform-container">
                    <div className="waveform-header">
                        <div id="waveform-time" className="waveform-header-item"></div>
                        <div id="waveform-region-time" className="waveform-header-item"></div>
                    </div>
                    <div id="waveform"></div>
                    <div id="wave-timeline"></div>
                </div>
                <div className="height-60 mel-spectrogram-container">
                    <MelSpectrogram></MelSpectrogram>
                </div>

            </>
        )
    }, [])

    return (
        <>
            {view}
        </>
    )
}