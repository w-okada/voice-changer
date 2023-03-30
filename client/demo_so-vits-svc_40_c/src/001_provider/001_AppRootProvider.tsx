import React, { useContext } from "react";
import { ReactNode } from "react";
import { AudioConfigState, useAudioConfig } from "../001_globalHooks/001_useAudioConfig";

type Props = {
    children: ReactNode;
};

type AppRootValue = {
    audioContextState: AudioConfigState

}

const AppRootContext = React.createContext<AppRootValue | null>(null);
export const useAppRoot = (): AppRootValue => {
    const state = useContext(AppRootContext);
    if (!state) {
        throw new Error("useAppState must be used within AppStateProvider");
    }
    return state;
};

export const AppRootProvider = ({ children }: Props) => {
    const audioContextState = useAudioConfig()
    const providerValue: AppRootValue = {
        audioContextState,
    };

    return <AppRootContext.Provider value={providerValue}>{children}</AppRootContext.Provider>;
};
