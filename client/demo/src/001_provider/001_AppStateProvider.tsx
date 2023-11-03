import { ClientState } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect, useRef } from "react";
import { ReactNode } from "react";
import { useVCClient } from "../001_globalHooks/001_useVCClient";
import { useAppRoot } from "./001_AppRootProvider";
import { useMessageBuilder } from "../hooks/useMessageBuilder";

import { VoiceChangerJSClient } from "./VoiceChangerJSClient";

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
    const messageBuilderState = useMessageBuilder();
    const voiceChangerJSClient = useRef<VoiceChangerJSClient>();

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

    // useEffect(() => {
    //     if (clientState.clientState.initialized) {
    //         voiceChangerJSClient.current = new VoiceChangerJSClient();
    //         voiceChangerJSClient.current.initialize();
    //         clientState.clientState.setInternalAudioProcessCallback({
    //             processAudio: async (data: Uint8Array) => {
    //                 console.log("[CLIENTJS] start --------------------------------------");
    //                 const audioF32 = new Float32Array(data.buffer);
    //                 const converted = await voiceChangerJSClient.current!.convert(audioF32);

    //                 let audio_int16_out = new Int16Array(converted.length);
    //                 for (let i = 0; i < converted.length; i++) {
    //                     audio_int16_out[i] = converted[i] * 32768.0;
    //                 }
    //                 const res = new Uint8Array(audio_int16_out.buffer);
    //                 console.log("AUDIO::::audio_int16_out", audio_int16_out);

    //                 console.log("[CLIENTJS] end --------------------------------------");
    //                 return res;
    //             },
    //         });
    //     }
    // }, [clientState.clientState.initialized]);

    const providerValue: AppStateValue = {
        audioContext: appRoot.audioContextState.audioContext!,
        ...clientState.clientState,
        initializedRef,
    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
