import { ClientState } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect, useRef } from "react";
import { ReactNode } from "react";
import { useVCClient } from "../001_globalHooks/001_useVCClient";
import { useAppRoot } from "./001_AppRootProvider";
import { toast } from "react-toastify";

type Props = {
    children: ReactNode;
};

type AppStateValue = ClientState & {
    audioContext: AudioContext;
    initializedRef: React.MutableRefObject<boolean>;
};

const AppStateContext = React.createContext<AppStateValue | null>(null);
export const useAppState = (): AppStateValue => {
    const state = useContext(AppStateContext);
    if (!state) {
        throw new Error("useAppState must be used within AppStateProvider");
    }
    return state;
};

export const AppStateProvider = ({ children }: Props) => {
    const appRoot = useAppRoot();
    const clientState = useVCClient({ audioContext: appRoot.audioContextState.audioContext });

    const initializedRef = useRef<boolean>(false);
    useEffect(() => {
        if (clientState.clientState.initialized) {
            initializedRef.current = true;
            clientState.clientState.getInfo();
            // clientState.clientState.setVoiceChangerClientSetting({
            //     ...clientState.clientState.setting.voiceChangerClientSetting
            // })
        }
    }, [clientState.clientState.initialized]);

    useEffect(() => {
        if (clientState.clientState.errorMessage) {
            toast.error(clientState.clientState.errorMessage);
            clientState.clientState.resetErrorMessage();
        }
    }, [clientState.clientState.errorMessage]);

    const providerValue: AppStateValue = {
        audioContext: appRoot.audioContextState.audioContext!,
        ...clientState.clientState,
        initializedRef,
    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
