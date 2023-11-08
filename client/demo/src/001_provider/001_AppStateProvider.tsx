import { ClientState } from "@dannadori/voice-changer-client-js";
import { VoiceChangerJSClient } from "@dannadori/voice-changer-js";
import React, { useContext, useEffect, useRef } from "react";
import { ReactNode } from "react";
import { useVCClient } from "../001_globalHooks/001_useVCClient";
import { useAppRoot } from "./001_AppRootProvider";
import { useMessageBuilder } from "../hooks/useMessageBuilder";

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
    //         const baseUrl = "https://192.168.0.247:18888";
    //         // const modelUrl = `${baseUrl}/models/rvc2v_40k_f0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvc2v_40k_nof0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvc2v_16k_f0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvcv2_amitaro_v2_40k_f0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvcv2_amitaro_v2_40k_nof0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvcv2_amitaro_v2_32k_f0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvcv2_amitaro_v2_32k_nof0_24000.bin`;

    //         // const modelUrl = `${baseUrl}/models/rvcv1_amitaro_v1_32k_f0_24000.bin`;
    //         const modelUrl = `${baseUrl}/models/rvcv1_amitaro_v1_32k_nof0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvcv1_amitaro_v1_40k_f0_24000.bin`;
    //         // const modelUrl = `${baseUrl}/models/rvcv1_amitaro_v1_40k_nof0_24000.bin`;

    //         voiceChangerJSClient.current = new VoiceChangerJSClient();
    //         voiceChangerJSClient.current.initialize(
    //             {
    //                 baseUrl: baseUrl,
    //                 inputSamplingRate: 48000,
    //                 outputSamplingRate: 48000,
    //             },
    //             modelUrl,
    //         );
    //         clientState.clientState.setInternalAudioProcessCallback({
    //             processAudio: async (data: Uint8Array) => {
    //                 const audioF32 = new Float32Array(data.buffer);
    //                 const converted = await voiceChangerJSClient.current!.convert(audioF32);
    //                 const res = new Uint8Array(converted.buffer);
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
