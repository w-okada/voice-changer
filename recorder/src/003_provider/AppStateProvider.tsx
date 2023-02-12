import React, { useContext } from "react";
import { ReactNode } from "react";
import { CorpusDataStateAndMethod, useCorpusData } from "../002_hooks/003_useCorpusData";
import { MediaRecorderStateAndMethod, useMediaRecorder } from "../002_hooks/012_useMediaRecorder";
import { AudioControllerStateAndMethod, useAudioControllerState } from "../002_hooks/013_useAudioControllerState";
import { useWaveSurfer, WaveSurferStateAndMethod } from "../002_hooks/014_useWaveSurfer";
import { FrontendManagerStateAndMethod, useFrontendManager } from "../002_hooks/100_useFrontendManager";

type Props = {
    children: ReactNode;
};

interface AppStateValue {
    mediaRecorderState: MediaRecorderStateAndMethod
    audioControllerState: AudioControllerStateAndMethod
    waveSurferState: WaveSurferStateAndMethod
    corpusDataState: CorpusDataStateAndMethod
    frontendManagerState: FrontendManagerStateAndMethod;
}

const AppStateContext = React.createContext<AppStateValue | null>(null);
export const useAppState = (): AppStateValue => {
    const state = useContext(AppStateContext);
    if (!state) {
        throw new Error("useAppState must be used within AppStateProvider");
    }
    return state;
};

export const AppStateProvider = ({ children }: Props) => {
    const corpusDataState = useCorpusData()
    const mediaRecorderState = useMediaRecorder()
    const audioControllerState = useAudioControllerState()
    const waveSurferState = useWaveSurfer()
    const frontendManagerState = useFrontendManager();

    const providerValue = {
        mediaRecorderState,
        audioControllerState,
        waveSurferState,
        corpusDataState,
        frontendManagerState
    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
