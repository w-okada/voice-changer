// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。
// 24000sample -> 1sec, 128sample(1chunk) -> 5.333msec
// 187.5chunk -> 1sec

export const VoiceChangerType = {
    RVC: "RVC",
} as const;
export type VoiceChangerType = (typeof VoiceChangerType)[keyof typeof VoiceChangerType];

///////////////////////
// サーバセッティング
///////////////////////
export const InputSampleRate = {
    "48000": 48000,
    "44100": 44100,
    "24000": 24000,
} as const;
export type InputSampleRate = (typeof InputSampleRate)[keyof typeof InputSampleRate];

export const ModelSamplingRate = {
    "48000": 48000,
    "40000": 40000,
    "32000": 32000,
} as const;
export type ModelSamplingRate = (typeof InputSampleRate)[keyof typeof InputSampleRate];

export const F0Detector = {
    dio: "dio",
    harvest: "harvest",
    crepe_full: "crepe_full",
    crepe_tiny: "crepe_tiny",
    crepe_full_onnx: "crepe_full_onnx",
    crepe_tiny_onnx: "crepe_tiny_onnx",
    rmvpe: "rmvpe",
    rmvpe_onnx: "rmvpe_onnx",
    fcpe: "fcpe",
    fcpe_onnx: "fcpe_onnx",
} as const;
export type F0Detector = (typeof F0Detector)[keyof typeof F0Detector];

export const DiffMethod = {
    pndm: "pndm",
    "dpm-solver": "dpm-solver",
} as const;
export type DiffMethod = (typeof DiffMethod)[keyof typeof DiffMethod];

export const RVCModelType = {
    pyTorchRVC: "pyTorchRVC",
    pyTorchRVCNono: "pyTorchRVCNono",
    pyTorchRVCv2: "pyTorchRVCv2",
    pyTorchRVCv2Nono: "pyTorchRVCv2Nono",
    pyTorchWebUI: "pyTorchWebUI",
    pyTorchWebUINono: "pyTorchWebUINono",
    onnxRVC: "onnxRVC",
    onnxRVCNono: "onnxRVCNono",
} as const;
export type RVCModelType = (typeof RVCModelType)[keyof typeof RVCModelType];

export const ServerSettingKey = {
    passThrough: "passThrough",
    srcId: "srcId",
    dstId: "dstId",
    gpu: "gpu",

    crossFadeOverlapSize: "crossFadeOverlapSize",

    framework: "framework",
    onnxExecutionProvider: "onnxExecutionProvider",

    f0Factor: "f0Factor",
    f0Detector: "f0Detector",
    recordIO: "recordIO",

    enableServerAudio: "enableServerAudio",
    serverAudioStated: "serverAudioStated",
    serverAudioSampleRate: "serverAudioSampleRate",
    serverInputAudioSampleRate: "serverInputAudioSampleRate",
    serverOutputAudioSampleRate: "serverOutputAudioSampleRate",
    serverMonitorAudioSampleRate: "serverMonitorAudioSampleRate",
    serverInputAudioBufferSize: "serverInputAudioBufferSize",
    serverOutputAudioBufferSize: "serverOutputAudioBufferSize",
    serverInputDeviceId: "serverInputDeviceId",
    serverOutputDeviceId: "serverOutputDeviceId",
    serverMonitorDeviceId: "serverMonitorDeviceId",
    serverReadChunkSize: "serverReadChunkSize",
    serverInputAudioGain: "serverInputAudioGain",
    serverOutputAudioGain: "serverOutputAudioGain",
    serverMonitorAudioGain: "serverMonitorAudioGain",

    tran: "tran",
    formantShift: "formantShift",
    silentThreshold: "silentThreshold",
    extraConvertSize: "extraConvertSize",

    indexRatio: "indexRatio",
    protect: "protect",
    forceFp32: "forceFp32",
    modelSamplingRate: "modelSamplingRate",
    silenceFront: "silenceFront",
    modelSlotIndex: "modelSlotIndex",

    inputSampleRate: "inputSampleRate",
} as const;
export type ServerSettingKey = (typeof ServerSettingKey)[keyof typeof ServerSettingKey];

export type VoiceChangerServerSetting = {
    passThrough: boolean;
    srcId: number;
    dstId: number;
    gpu: number;

    crossFadeOverlapSize: number;

    f0Factor: number;
    f0Detector: F0Detector; // dio or harvest
    recordIO: number; // 0:off, 1:on

    enableServerAudio: number; // 0:off, 1:on
    serverAudioStated: number; // 0:off, 1:on
    serverAudioSampleRate: number;
    serverInputAudioSampleRate: number;
    serverOutputAudioSampleRate: number;
    serverMonitorAudioSampleRate: number;
    serverInputAudioBufferSize: number;
    serverOutputAudioBufferSize: number;
    serverInputDeviceId: number;
    serverOutputDeviceId: number;
    serverMonitorDeviceId: number;
    serverReadChunkSize: number;
    serverInputAudioGain: number;
    serverOutputAudioGain: number;
    serverMonitorAudioGain: number;

    tran: number;
    formantShift: number;
    silentThreshold: number;
    extraConvertSize: number;

    indexRatio: number; // RVC
    protect: number; // RVC
    silenceFront: number; // 0:off, 1:on
    forceFp32: number; // 0:off, 1:on
    modelSamplingRate: ModelSamplingRate; // 32000,40000,48000
    modelSlotIndex: number;

    inputSampleRate: InputSampleRate;
};

type ModelSlot = {
    slotIndex: number;
    voiceChangerType: VoiceChangerType;
    name: string;
    description: string;
    credit: string;
    termsOfUseUrl: string;
    iconFile: string;
    speakers: { [key: number]: string };
};

export type RVCModelSlot = ModelSlot & {
    modelFile: string;
    indexFile: string;
    defaultIndexRatio: number;
    defaultProtect: number;
    defaultTune: number;
    defaultFormantShift: number;
    modelType: RVCModelType;

    embChannels: number;
    f0: boolean;
    samplingRate: number;
    deprecated: boolean;
};

export type ModelSlotUnion = RVCModelSlot;

type ServerAudioDevice = {
    kind: "audioinput" | "audiooutput";
    index: number;
    name: string;
    hostAPI: string;
};

export type ServerInfo = VoiceChangerServerSetting & {
    // コンフィグ対象外 (getInfoで取得のみ可能な情報)
    status: string;
    modelSlots: ModelSlotUnion[];
    serverAudioInputDevices: ServerAudioDevice[];
    serverAudioOutputDevices: ServerAudioDevice[];
    sampleModels: RVCSampleModel[];
    gpus: {
        id: number;
        name: string;
        memory: number;
    }[];
    maxInputLength: number; // MMVCv15
    voiceChangerParams: {
        model_dir: string;
    };
};

export type SampleModel = {
    id: string;
    voiceChangerType: VoiceChangerType;
    lang: string;
    tag: string[];
    name: string;
    modelUrl: string;
    termsOfUseUrl: string;
    icon: string;
    credit: string;
    description: string;
    sampleRate: number;
    modelType: string;
    f0: boolean;
};

export type RVCSampleModel = SampleModel & {
    indexUrl: string;
    featureUrl: string;
};

export const DefaultServerSetting: ServerInfo = {
    // VC Common
    passThrough: false,
    inputSampleRate: 48000,

    crossFadeOverlapSize: 0.10,

    recordIO: 0,

    enableServerAudio: 0,
    serverAudioStated: 0,
    serverAudioSampleRate: 48000,
    serverInputAudioSampleRate: 48000,
    serverOutputAudioSampleRate: 48000,
    serverMonitorAudioSampleRate: 48000,
    serverInputAudioBufferSize: 1024 * 24,
    serverOutputAudioBufferSize: 1024 * 24,
    serverInputDeviceId: -1,
    serverOutputDeviceId: -1,
    serverMonitorDeviceId: -1,
    serverReadChunkSize: 192,
    serverInputAudioGain: 1.0,
    serverOutputAudioGain: 1.0,
    serverMonitorAudioGain: 1.0,

    // VC Specific
    srcId: 0,
    dstId: 1,
    gpu: 0,

    f0Factor: 1.0,
    f0Detector: F0Detector.rmvpe_onnx,

    tran: 0,
    formantShift: 0,
    silentThreshold: 0,
    extraConvertSize: 0,

    indexRatio: 0,
    protect: 0.5,
    modelSamplingRate: 48000,
    silenceFront: 1,
    forceFp32: 0,
    modelSlotIndex: 0,
    sampleModels: [],
    gpus: [],

    //
    status: "ok",
    modelSlots: [],
    serverAudioInputDevices: [],
    serverAudioOutputDevices: [],

    maxInputLength: 128 * 2048,
    voiceChangerParams: {
        model_dir: "",
    },
};

///////////////////////
// Workletセッティング
///////////////////////

export type WorkletSetting = {
};
///////////////////////
// Worklet Nodeセッティング
///////////////////////
export const Protocol = {
    sio: "sio",
    rest: "rest"
} as const;
export type Protocol = (typeof Protocol)[keyof typeof Protocol];

export const SendingSampleRate = {
    "48000": 48000,
    "44100": 44100,
    "24000": 24000,
} as const;
export type SendingSampleRate = (typeof SendingSampleRate)[keyof typeof SendingSampleRate];

export const DownSamplingMode = {
    decimate: "decimate",
    average: "average",
} as const;
export type DownSamplingMode = (typeof DownSamplingMode)[keyof typeof DownSamplingMode];

export type WorkletNodeSetting = {
    serverUrl: string;
    protocol: Protocol;
    sendingSampleRate: SendingSampleRate;
    inputChunkNum: number;
    downSamplingMode: DownSamplingMode;
};

///////////////////////
// クライアントセッティング
///////////////////////
export const SampleRate = {
    "48000": 48000,
} as const;
export type SampleRate = (typeof SampleRate)[keyof typeof SampleRate];

export type VoiceChangerClientSetting = {
    audioInput: string | MediaStream | null;
    sampleRate: SampleRate; // 48000Hz
    echoCancel: boolean;
    noiseSuppression: boolean;
    noiseSuppression2: boolean;

    inputGain: number;
    outputGain: number;
    monitorGain: number;

    passThroughConfirmationSkip: boolean;
};

///////////////////////
// Client セッティング
///////////////////////
export type ClientSetting = {
    workletSetting: WorkletSetting;
    workletNodeSetting: WorkletNodeSetting;
    voiceChangerClientSetting: VoiceChangerClientSetting;
};
export const DefaultClientSettng: ClientSetting = {
    workletSetting: {
    },
    workletNodeSetting: {
        serverUrl: "",
        protocol: "sio",
        sendingSampleRate: 48000,
        inputChunkNum: 192,
        downSamplingMode: "average",
    },
    voiceChangerClientSetting: {
        audioInput: null,
        sampleRate: 48000,
        echoCancel: false,
        noiseSuppression: false,
        noiseSuppression2: false,
        inputGain: 1.0,
        outputGain: 1.0,
        monitorGain: 1.0,
        passThroughConfirmationSkip: false,
    },
};

////////////////////////////////////
// Exceptions
////////////////////////////////////
export const VOICE_CHANGER_CLIENT_EXCEPTION = {
    ERR_SIO_CONNECT_FAILED: "ERR_SIO_CONNECT_FAILED",
    ERR_SIO_INVALID_RESPONSE: "ERR_SIO_INVALID_RESPONSE",
    ERR_REST_INVALID_RESPONSE: "ERR_REST_INVALID_RESPONSE",
    ERR_MIC_STREAM_NOT_INITIALIZED: "ERR_MIC_STREAM_NOT_INITIALIZED",
    ERR_INTERNAL_AUDIO_PROCESS_CALLBACK_IS_NOT_INITIALIZED: "ERR_INTERNAL_AUDIO_PROCESS_CALLBACK_IS_NOT_INITIALIZED",
    ERR_GENERIC_VOICE_CHANGER_EXCEPTION: "ERR_GENERIC_VOICE_CHANGER_EXCEPTION",
} as const;
export type VOICE_CHANGER_CLIENT_EXCEPTION = (typeof VOICE_CHANGER_CLIENT_EXCEPTION)[keyof typeof VOICE_CHANGER_CLIENT_EXCEPTION];

////////////////////////////////////
// indexedDB
////////////////////////////////////
export const INDEXEDDB_DB_APP_NAME = "INDEXEDDB_KEY_VOICE_CHANGER";
export const INDEXEDDB_DB_NAME = "INDEXEDDB_KEY_VOICE_CHANGER_DB";
export const INDEXEDDB_KEY_CLIENT = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_CLIENT";
export const INDEXEDDB_KEY_SERVER = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_SERVER";
export const INDEXEDDB_KEY_MODEL_DATA = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_MODEL_DATA";

// ONNX
export type OnnxExporterInfo = {
    status: string;
    path: string;
    filename: string;
};

// Merge
export type MergeElement = {
    slotIndex: number;
    strength: number;
};
export type MergeModelRequest = {
    voiceChangerType: VoiceChangerType;
    command: "mix";
    files: MergeElement[];
};
