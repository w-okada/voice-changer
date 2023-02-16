import { ClientState } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect } from "react";
import { ReactNode } from "react";
import { useVCClient, VCClientState } from "../001_globalHooks/001_useVCClient";
import { FrontendManagerStateAndMethod, useFrontendManager } from "../001_globalHooks/010_useFrontendManager";
import { useAppRoot } from "./001_AppRootProvider";

type Props = {
    children: ReactNode;
};

type AppStateValue = ClientState & {
    audioContext: AudioContext
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
    const appRoot = useAppRoot()
    const clientState = useVCClient({ audioContext: appRoot.audioContextState.audioContext! })
    const frontendManagerState = useFrontendManager();

    const providerValue: AppStateValue = {
        audioContext: appRoot.audioContextState.audioContext!,
        ...clientState.clientState,
        frontendManagerState
    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
