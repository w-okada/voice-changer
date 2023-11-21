import { ClientState, WebModelSlot } from "@dannadori/voice-changer-client-js";
import { VoiceChangerJSClientConfig, VoiceChangerJSClient, ProgressUpdateType, ProgreeeUpdateCallbcckInfo, VoiceChangerType, InputLengthKey } from "@dannadori/voice-changer-js";
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
    upkey: number;
};
export type WebInfoStateAndMethod = WebInfoState & {
    loadVoiceChanagerModel: () => Promise<void>;
    setUpkey: (upkey: number) => void;
};

const ModelSampleRateStr = {
    "40k": "40k",
    "32k": "32k",
} as const;
type ModelSampleRateStr = (typeof ModelSampleRateStr)[keyof typeof ModelSampleRateStr];

const noF0ModelUrl: { [modelType in VoiceChangerType]: { [inputLength in InputLengthKey]: { [sampleRate in ModelSampleRateStr]: string } } } = {
    rvcv1: {
        "24000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_24000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_24000.bin",
        },
        "16000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_16000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_16000.bin",
        },
        "12000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_12000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_12000.bin",
        },
        "8000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_nof0_8000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_nof0_8000.bin",
        },
    },
    rvcv2: {
        "24000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_24000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_24000.bin",
        },
        "16000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_16000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_16000.bin",
        },
        "12000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_12000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_12000.bin",
        },
        "8000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_nof0_8000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_nof0_8000.bin",
        },
    },
};
const f0ModelUrl: { [modelType in VoiceChangerType]: { [inputLength in InputLengthKey]: { [sampleRate in ModelSampleRateStr]: string } } } = {
    rvcv1: {
        "24000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_24000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_24000.bin",
        },
        "16000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_16000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_16000.bin",
        },
        "12000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_12000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_12000.bin",
        },
        "8000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_40k_f0_8000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv1_amitaro_v1_32k_f0_8000.bin",
        },
    },
    rvcv2: {
        "24000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_24000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_24000.bin",
        },
        "16000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_16000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_16000.bin",
        },
        "12000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_12000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_12000.bin",
        },
        "8000": {
            "40k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_40k_f0_8000.bin",
            "32k": "https://huggingface.co/wok000/vcclient_model/resolve/main/web_model/v_01_alpha/amitaro/rvcv2_amitaro_v2_32k_f0_8000.bin",
        },
    },
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

export const useWebInfo = (props: UseWebInfoProps): WebInfoStateAndMethod => {
    const voiceChangerType: VoiceChangerType = "rvcv1";
    const inputLength: InputLengthKey = "24000";
    const useF0 = false;
    const sampleRate: ModelSampleRateStr = "40k";
    let modelUrl;
    if (useF0) {
        modelUrl = f0ModelUrl[voiceChangerType][inputLength][sampleRate];
    } else {
        modelUrl = noF0ModelUrl[voiceChangerType][inputLength][sampleRate];
    }
    const [voiceChangerConfig, setVoiceChangerConfig] = useState<VoiceChangerConfig>({
        config: {
            voiceChangerType: voiceChangerType,
            inputLength: inputLength,
            baseUrl: window.location.origin,
            inputSamplingRate: 48000,
            outputSamplingRate: 48000,
        },
        modelUrl: modelUrl,
        progressCallback: null,
        portrait: `${window.location.origin}/models/amitaro.png`,
        name: "あみたろ",
        termOfUse: "https://huggingface.co/wok000/vcclient_model/raw/main/rvc/amitaro_contentvec_256/term_of_use.txt",
        f0: false,
    });
    const [webModelLoadingState, setWebModelLoadingState] = useState<WebModelLoadingState>(WebModelLoadingState.none);
    const [progressLoadPreprocess, setProgressLoadPreprocess] = useState<number>(0);
    const [progressLoadVCModel, setProgressLoadVCModel] = useState<number>(0);
    const [progressWarmup, setProgressWarmup] = useState<number>(0);
    const [upkey, setUpkey] = useState<number>(0);
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
        const warmupResult = await voiceChangerJSClient.current.checkResponseTime();
        console.log("warmup result", warmupResult);

        // check time
        const responseTimeInfo = await voiceChangerJSClient.current.checkResponseTime();
        console.log("responseTimeInfo", responseTimeInfo);

        props.clientState?.setInternalAudioProcessCallback({
            processAudio: async (data: Uint8Array) => {
                const audioF32 = new Float32Array(data.buffer);
                const res = await voiceChangerJSClient.current!.convert(audioF32);
                const audio = new Uint8Array(res[0].buffer);
                if (res[1]) {
                    console.log("RESPONSE!", res[1]);
                }
                return audio;
            },
        });
        setWebModelLoadingState("ready");
    };
    useEffect(() => {
        if (!voiceChangerJSClient.current) {
            console.log("setupkey", voiceChangerJSClient.current);
            return;
        }
        voiceChangerJSClient.current.setUpkey(upkey);
    }, [upkey]);

    return {
        voiceChangerConfig,
        webModelLoadingState,
        progressLoadPreprocess,
        progressLoadVCModel,
        progressWarmup,
        webModelslot,
        upkey,
        loadVoiceChanagerModel,
        setUpkey,
    };
};
