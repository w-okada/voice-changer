import { ClientState } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect, useRef } from "react";
import { ReactNode } from "react";
import { AppSettingStates, useAppSettings } from "../001_globalHooks/001_useAppSettings";
import { useVCClient } from "../001_globalHooks/001_useVCClient";
import { FrontendManagerStateAndMethod, useFrontendManager } from "../001_globalHooks/010_useFrontendManager";
import { PsdAnimationStateAndMethod, usePsdAnimation } from "../001_globalHooks/100_usePsdAnimation";
import { useAppRoot } from "./001_AppRootProvider";

type Props = {
    children: ReactNode;
};

type AppStateValue = ClientState & {
    audioContext: AudioContext
    appSettings: AppSettingStates
    frontendManagerState: FrontendManagerStateAndMethod;
    initializedRef: React.MutableRefObject<boolean>
    psdAnimationState: PsdAnimationStateAndMethod
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
    const appSettings = useAppSettings()
    const clientState = useVCClient({ audioContext: appRoot.audioContextState.audioContext })
    const frontendManagerState = useFrontendManager();
    const psdAnimationState = usePsdAnimation()

    const initializedRef = useRef<boolean>(false)
    useEffect(() => {
        if (clientState.clientState.initialized) {
            initializedRef.current = true
        }
    }, [clientState.clientState.initialized])

    useEffect(() => {
        if (appSettings.appSettings.charaName.length > 0) {
            psdAnimationState.loadPsd(appSettings.appSettings.psdFile, appSettings.appSettings.motionFile)
        }
    }, [appSettings.appSettings])


    const providerValue: AppStateValue = {
        audioContext: appRoot.audioContextState.audioContext!,
        appSettings,
        ...clientState.clientState,
        frontendManagerState,
        psdAnimationState,
        initializedRef


    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
