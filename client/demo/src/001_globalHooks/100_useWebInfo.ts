import { ClientState, WebModelSlot } from "@dannadori/voice-changer-client-js";
import { VoiceChangerJSClientConfig, VoiceChangerJSClient, ProgressUpdateType, ProgreeeUpdateCallbcckInfo } from "@dannadori/voice-changer-js";
import { useEffect, useMemo, useRef, useState } from "react";

export type UseWebInfoProps = {
    clientState: ClientState | null;
};

export const WebModelLoadingState = {
    none: "none",
    loading: "loading",
    warmup: "warmup",
    ready: "ready",
} as const;
export type WebModelLoadingState = (typeof WebModelLoadingState)[keyof typeof WebModelLoadingState];

export type VoiceChangerConfig = {
    config: VoiceChangerJSClientConfig;
    modelUrl: string;
    progressCallback?: ((data: any) => void) | null;
    portrait: string;
    name: string;
    termOfUse: string;
    f0: boolean;
};
export type WebInfoState = {
    voiceChangerConfig: VoiceChangerConfig;
    webModelLoadingState: WebModelLoadingState;
    progressLoadPreprocess: number;
    progressLoadVCModel: number;
    progressWarmup: number;
    webModelslot: WebModelSlot;
};
export type WebInfoStateAndMethod = WebInfoState & {
    loadVoiceChanagerModel: () => Promise<void>;
};

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_24000.bin

// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_8000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_12000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_16000.bin
// https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_24000.bin

const InitialVoiceChangerConfig: VoiceChangerConfig = {
    config: {
        voiceChangerType: "rvcv1",
        inputLength: "24000",
        baseUrl: window.location.origin,
        inputSamplingRate: 48000,
        outputSamplingRate: 48000,
    },
    // modelUrl: `${window.location.origin}/models/rvcv1_amitaro_v1_32k_nof0_24000.bin`,
    modelUrl: `https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_24000.bin`,
    progressCallback: null,
    portrait: `${window.location.origin}/models/amitaro.png`,
    name: "あみたろ",
    termOfUse: "https://huggingface.co/wok000/vcclient_model/raw/main/rvc/amitaro_contentvec_256/term_of_use.txt",
    f0: false,
};

export const useWebInfo = (props: UseWebInfoProps): WebInfoStateAndMethod => {
    const [voiceChangerConfig, setVoiceChangerConfig] = useState<VoiceChangerConfig>(InitialVoiceChangerConfig);
    const [webModelLoadingState, setWebModelLoadingState] = useState<WebModelLoadingState>(WebModelLoadingState.none);
    const [progressLoadPreprocess, setProgressLoadPreprocess] = useState<number>(0);
    const [progressLoadVCModel, setProgressLoadVCModel] = useState<number>(0);
    const [progressWarmup, setProgressWarmup] = useState<number>(0);
    const voiceChangerJSClient = useRef<VoiceChangerJSClient>();

    const webModelslot: WebModelSlot = useMemo(() => {
        return {
            slotIndex: -1,
            voiceChangerType: "WebModel",
            name: voiceChangerConfig.name,
            description: "",
            credit: "",
            termsOfUseUrl: voiceChangerConfig.termOfUse,
            iconFile: voiceChangerConfig.portrait,
            speakers: {},
            defaultTune: 0,
            modelType: "pyTorchRVCNono",
            f0: voiceChangerConfig.f0,
            samplingRate: 0,
            modelFile: "",
        };
    }, []);

    useEffect(() => {
        const progressCallback = (data: ProgreeeUpdateCallbcckInfo) => {
            if (data.progressUpdateType === ProgressUpdateType.loadPreprocessModel) {
                setProgressLoadPreprocess(data.progress);
            } else if (data.progressUpdateType === ProgressUpdateType.loadVCModel) {
                setProgressLoadVCModel(data.progress);
            } else if (data.progressUpdateType === ProgressUpdateType.checkResponseTime) {
                setProgressWarmup(data.progress);
            }
        };
        setVoiceChangerConfig({ ...voiceChangerConfig, progressCallback });
    }, []);

    const loadVoiceChanagerModel = async () => {
        if (!props.clientState) {
            throw new Error("[useWebInfo] clientState is null");
        }
        if (!props.clientState.initialized) {
            console.warn("[useWebInfo] clientState is not initialized yet");
            return;
        }
        setWebModelLoadingState("loading");
        voiceChangerJSClient.current = new VoiceChangerJSClient();
        await voiceChangerJSClient.current.initialize(voiceChangerConfig.config, voiceChangerConfig.modelUrl, voiceChangerConfig.progressCallback);

        // worm up
        setWebModelLoadingState("warmup");
        const warmupResult = await voiceChangerJSClient.current.checkResponseTime((progress: number) => {
            console.log(`Recieve Progress: ${progress}`);
        });
        console.log("warmup result", warmupResult);

        // check time
        const responseTimeInfo = await voiceChangerJSClient.current.checkResponseTime();
        console.log("responseTimeInfo", responseTimeInfo);

        props.clientState?.setInternalAudioProcessCallback({
            processAudio: async (data: Uint8Array) => {
                const audioF32 = new Float32Array(data.buffer);
                const res = await voiceChangerJSClient.current!.convert(audioF32);
                const audio = new Uint8Array(res[0].buffer);
                console.log("RESPONSE!", res[1]);
                return audio;
            },
        });
        setWebModelLoadingState("ready");
    };

    return {
        voiceChangerConfig,
        webModelLoadingState,
        progressLoadPreprocess,
        progressLoadVCModel,
        progressWarmup,
        webModelslot,
        loadVoiceChanagerModel,
    };
};
