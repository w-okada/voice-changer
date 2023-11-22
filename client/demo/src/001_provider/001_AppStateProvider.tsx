import { ClientState } from "@dannadori/voice-changer-client-js";
import { VoiceChangerJSClient } from "@dannadori/voice-changer-js";
import React, { useContext, useEffect, useRef } from "react";
import { ReactNode } from "react";
import { useVCClient } from "../001_globalHooks/001_useVCClient";
import { useAppRoot } from "./001_AppRootProvider";
import { useMessageBuilder } from "../hooks/useMessageBuilder";
import { WebInfoStateAndMethod, useWebInfo } from "../001_globalHooks/100_useWebInfo";

type Props = {
    children: ReactNode;
};

type AppStateValue = ClientState & {
    audioContext: AudioContext;
    initializedRef: React.MutableRefObject<boolean>;
    webInfoState: WebInfoStateAndMethod;
    webEdition: boolean;
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
    const webEdition = appRoot.appGuiSettingState.edition.indexOf("web") >= 0;
    const clientState = useVCClient({ audioContext: appRoot.audioContextState.audioContext });
    const messageBuilderState = useMessageBuilder();
    const webInfoState = useWebInfo({ clientState: clientState.clientState, webEdition: webEdition });
    // const voiceChangerJSClient = useRef<VoiceChangerJSClient>();

    useEffect(() => {
        messageBuilderState.setMessage(__filename, "ioError", {
            ja: "エラーが頻発しています。対象としているフレームワークのモデルがロードされているか確認してください。",
            en: "Frequent errors occur. Please check if the model of the framework being targeted is loaded.",
        });
    }, []);

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
        if (clientState.clientState.ioErrorCount > 100) {
            alert(messageBuilderState.getMessage(__filename, "ioError"));
            clientState.clientState.resetIoErrorCount();
        }
    }, [clientState.clientState.ioErrorCount]);

    useEffect(() => {
        if (appRoot.appGuiSettingState.edition.indexOf("web") >= 0 && clientState.clientState.initialized) {
            clientState.clientState.setWorkletNodeSetting({ ...clientState.clientState.setting.workletNodeSetting, protocol: "internal" });
            // webInfoState.loadVoiceChanagerModel(); // hook内でuseEffectでinvoke
        }
    }, [clientState.clientState.initialized]);

    const providerValue: AppStateValue = {
        audioContext: appRoot.audioContextState.audioContext!,
        ...clientState.clientState,
        initializedRef,
        webInfoState,
        webEdition,
    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
