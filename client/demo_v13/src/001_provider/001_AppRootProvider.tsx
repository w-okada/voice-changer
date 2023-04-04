import React, { useContext, useEffect } from "react";
import { ReactNode } from "react";
import { AppGuiSettingStateAndMethod, userAppGuiSetting } from "../001_globalHooks/001_useAppGuiSetting";
import { AudioConfigState, useAudioConfig } from "../001_globalHooks/001_useAudioConfig";

type Props = {
    children: ReactNode;
};

type AppRootValue = {
    audioContextState: AudioConfigState
    appGuiSettingState: AppGuiSettingStateAndMethod
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
    const appGuiSettingState = userAppGuiSetting()

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const modelType = params.get("modelType") || ""
        appGuiSettingState.getAppSetting(`/assets/gui_settings/${modelType}.json`)
    }, [])

    const providerValue: AppRootValue = {
        audioContextState,
        appGuiSettingState
    };
    return <AppRootContext.Provider value={providerValue}>{children}</AppRootContext.Provider>;
};
