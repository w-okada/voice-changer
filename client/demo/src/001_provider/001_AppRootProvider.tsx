import { ClientType } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect, useState } from "react";
import { ReactNode } from "react";
import { AppGuiSettingStateAndMethod, useAppGuiSetting } from "../001_globalHooks/001_useAppGuiSetting";
import { AudioConfigState, useAudioConfig } from "../001_globalHooks/001_useAudioConfig";

type Props = {
    children: ReactNode;
};

type AppRootValue = {
    audioContextState: AudioConfigState
    appGuiSettingState: AppGuiSettingStateAndMethod
    clientType: ClientType | null
    setClientType: (val: ClientType | null) => void
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
    const appGuiSettingState = useAppGuiSetting()
    const [clientType, setClientType] = useState<ClientType | null>(null)

    useEffect(() => {
        if (!clientType) {
            return
        }
        appGuiSettingState.getAppGuiSetting(`/assets/gui_settings/${clientType}.json`)
    }, [clientType])

    const providerValue: AppRootValue = {
        audioContextState,
        appGuiSettingState,
        clientType,
        setClientType
    };
    return <AppRootContext.Provider value={providerValue}>{children}</AppRootContext.Provider>;
};
