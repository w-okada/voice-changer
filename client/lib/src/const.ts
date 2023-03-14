
// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。
// 24000sample -> 1sec, 128sample(1chunk) -> 5.333msec
// 187.5chunk -> 1sec

export const ClientType = {
    "MMVCv15": "MMVCv15",
    "MMVCv13": "MMVCv13",
    "so_vits_svc_40v2": "so_vits_svc_40v2",
    "so_vits_svc_40v2_tsukuyomi": "so_vits_svc_40v2c_tsukuyomi",

} as const
export type ClientType = typeof ClientType[keyof typeof ClientType]

///////////////////////
// サーバセッティング
///////////////////////
export const InputSampleRate = {
    "48000": 48000,
    "44100": 44100,
    "24000": 24000
} as const
export type InputSampleRate = typeof InputSampleRate[keyof typeof InputSampleRate]

export const CrossFadeOverlapSize = {
    "1024": 1024,
    "2048": 2048,
    "4096": 4096,
} as const
export type CrossFadeOverlapSize = typeof CrossFadeOverlapSize[keyof typeof CrossFadeOverlapSize]


export const OnnxExecutionProvider = {
    "CPUExecutionProvider": "CPUExecutionProvider",
    "CUDAExecutionProvider": "CUDAExecutionProvider",
    "DmlExecutionProvider": "DmlExecutionProvider",
    "OpenVINOExecutionProvider": "OpenVINOExecutionProvider",
} as const
export type OnnxExecutionProvider = typeof OnnxExecutionProvider[keyof typeof OnnxExecutionProvider]

export const Framework = {
    "PyTorch": "PyTorch",
    "ONNX": "ONNX",
} as const
export type Framework = typeof Framework[keyof typeof Framework]

export const F0Detector = {
    "dio": "dio",
    "harvest": "harvest",
} as const
export type F0Detector = typeof F0Detector[keyof typeof F0Detector]



export const ServerSettingKey = {
    "srcId": "srcId",
    "dstId": "dstId",
    "gpu": "gpu",

    "crossFadeOffsetRate": "crossFadeOffsetRate",
    "crossFadeEndRate": "crossFadeEndRate",
    "crossFadeOverlapSize": "crossFadeOverlapSize",

    "framework": "framework",
    "onnxExecutionProvider": "onnxExecutionProvider",

    "f0Factor": "f0Factor",
    "f0Detector": "f0Detector",
    "recordIO": "recordIO",

    "tran": "tran",
    "noiceScale": "noiceScale",
    "predictF0": "predictF0",
    "silentThreshold": "silentThreshold",
    "extraConvertSize": "extraConvertSize",
    "clusterInferRatio": "clusterInferRatio",

    "inputSampleRate": "inputSampleRate",
} as const
export type ServerSettingKey = typeof ServerSettingKey[keyof typeof ServerSettingKey]


export type VoiceChangerServerSetting = {
    srcId: number,
    dstId: number,
    gpu: number,

    crossFadeOffsetRate: number,
    crossFadeEndRate: number,
    crossFadeOverlapSize: CrossFadeOverlapSize,

    framework: Framework
    onnxExecutionProvider: OnnxExecutionProvider,

    f0Factor: number
    f0Detector: F0Detector // dio or harvest
    recordIO: number // 0:off, 1:on

    tran: number // so-vits-svc
    noiceScale: number // so-vits-svc
    predictF0: number // so-vits-svc
    silentThreshold: number // so-vits-svc
    extraConvertSize: number// so-vits-svc
    clusterInferRatio: number // so-vits-svc

    inputSampleRate: InputSampleRate
}

export type ServerInfo = VoiceChangerServerSetting & {
    status: string
    configFile: string,
    pyTorchModelFile: string,
    onnxModelFile: string,
    onnxExecutionProviders: OnnxExecutionProvider[]
}

export type ServerInfoSoVitsSVC = ServerInfo & {
    speakers: { [key: string]: number }
}
export const DefaultServerSetting_MMVCv15: ServerInfo = {
    srcId: 0,
    dstId: 101,
    gpu: 0,

    crossFadeOffsetRate: 0.0,
    crossFadeEndRate: 1.0,
    crossFadeOverlapSize: CrossFadeOverlapSize[1024],

    framework: Framework.PyTorch,
    f0Factor: 1.0,
    onnxExecutionProvider: OnnxExecutionProvider.CPUExecutionProvider,
    f0Detector: F0Detector.dio,
    recordIO: 0,

    tran: 0,
    noiceScale: 0,
    predictF0: 0,
    silentThreshold: 0,
    extraConvertSize: 0,
    clusterInferRatio: 0,

    inputSampleRate: 24000,

    // 
    status: "ok",
    configFile: "",
    pyTorchModelFile: "",
    onnxModelFile: "",
    onnxExecutionProviders: []
}

export const DefaultServerSetting_MMVCv13: ServerInfo = {
    srcId: 107,
    dstId: 100,
    gpu: 0,

    crossFadeOffsetRate: 0.0,
    crossFadeEndRate: 1.0,
    crossFadeOverlapSize: CrossFadeOverlapSize[1024],

    framework: Framework.ONNX,
    f0Factor: 1.0,
    onnxExecutionProvider: OnnxExecutionProvider.CPUExecutionProvider,
    f0Detector: F0Detector.dio,
    recordIO: 0,

    tran: 0,
    noiceScale: 0,
    predictF0: 0,
    silentThreshold: 0,
    extraConvertSize: 0,
    clusterInferRatio: 0,

    inputSampleRate: 24000,

    // 
    status: "ok",
    configFile: "",
    pyTorchModelFile: "",
    onnxModelFile: "",
    onnxExecutionProviders: []
}

export const DefaultServerSetting_so_vits_svc_40v2: ServerInfo = {
    srcId: 0,
    dstId: 0,
    gpu: 0,

    crossFadeOffsetRate: 0.0,
    crossFadeEndRate: 1.0,
    crossFadeOverlapSize: CrossFadeOverlapSize[1024],

    framework: Framework.PyTorch,
    f0Factor: 1.0,
    onnxExecutionProvider: OnnxExecutionProvider.CPUExecutionProvider,
    f0Detector: F0Detector.dio,
    recordIO: 0,

    // tran: 0,
    // noiceScale: 0,
    // predictF0: 0,
    // silentThreshold: 0,
    tran: 10,
    noiceScale: 0.3,
    predictF0: 0,
    silentThreshold: 0.00001,
    extraConvertSize: 1024 * 32,
    clusterInferRatio: 0.1,

    inputSampleRate: 24000,

    // 
    status: "ok",
    configFile: "",
    pyTorchModelFile: "",
    onnxModelFile: "",
    onnxExecutionProviders: []
}


///////////////////////
// Workletセッティング
///////////////////////

export type WorkletSetting = {
    numTrancateTreshold: number,
    volTrancateThreshold: number,
    volTrancateLength: number
}
export const DefaultWorkletSetting: WorkletSetting = {
    numTrancateTreshold: 100,
    volTrancateThreshold: 0.0005,
    volTrancateLength: 32
}
///////////////////////
// Worklet Nodeセッティング
///////////////////////
export const Protocol = {
    "sio": "sio",
    "rest": "rest",
} as const
export type Protocol = typeof Protocol[keyof typeof Protocol]

export const SendingSampleRate = {
    "48000": 48000,
    "44100": 44100,
    "24000": 24000
} as const
export type SendingSampleRate = typeof SendingSampleRate[keyof typeof SendingSampleRate]

export const DownSamplingMode = {
    "decimate": "decimate",
    "average": "average"
} as const
export type DownSamplingMode = typeof DownSamplingMode[keyof typeof DownSamplingMode]


export type WorkletNodeSetting = {
    serverUrl: string,
    protocol: Protocol,
    sendingSampleRate: SendingSampleRate,
    inputChunkNum: number,
    downSamplingMode: DownSamplingMode,
}
export const DefaultWorkletNodeSetting: WorkletNodeSetting = {
    serverUrl: "",
    protocol: "sio",
    sendingSampleRate: 24000,
    inputChunkNum: 48,
    downSamplingMode: "average"
}

export const DefaultWorkletNodeSetting_so_vits_svc_40v2: WorkletNodeSetting = {
    serverUrl: "",
    protocol: "sio",
    sendingSampleRate: 24000,
    inputChunkNum: 128,
    downSamplingMode: "average"
}

///////////////////////
// クライアントセッティング
///////////////////////
export const SampleRate = {
    "48000": 48000,
} as const
export type SampleRate = typeof SampleRate[keyof typeof SampleRate]

export type Speaker = {
    "id": number,
    "name": string,
}
export type Correspondence = {
    "sid": number,
    "correspondence": number,
    "dirname": string
}
export type VoiceChangerClientSetting = {
    audioInput: string | MediaStream | null,
    sampleRate: SampleRate, // 48000Hz
    echoCancel: boolean,
    noiseSuppression: boolean,
    noiseSuppression2: boolean

    speakers: Speaker[],
    correspondences: Correspondence[],
    inputGain: number
    outputGain: number
}

export const DefaultVoiceChangerClientSetting: VoiceChangerClientSetting = {
    audioInput: null,
    sampleRate: 48000,
    speakers: [
        {
            "id": 0,
            "name": "user"
        },
        {
            "id": 101,
            "name": "ずんだもん"
        },
        {
            "id": 102,
            "name": "そら"
        },
        {
            "id": 103,
            "name": "めたん"
        },
        {
            "id": 104,
            "name": "つむぎ"
        }
    ],
    correspondences: [],
    echoCancel: false,
    noiseSuppression: false,
    noiseSuppression2: false,
    inputGain: 1.0,
    outputGain: 1.0
}


////////////////////////////////////
// Exceptions
////////////////////////////////////
export const VOICE_CHANGER_CLIENT_EXCEPTION = {
    ERR_SIO_CONNECT_FAILED: "ERR_SIO_CONNECT_FAILED",
    ERR_SIO_INVALID_RESPONSE: "ERR_SIO_INVALID_RESPONSE",
    ERR_REST_INVALID_RESPONSE: "ERR_REST_INVALID_RESPONSE",
    ERR_MIC_STREAM_NOT_INITIALIZED: "ERR_MIC_STREAM_NOT_INITIALIZED"

} as const
export type VOICE_CHANGER_CLIENT_EXCEPTION = typeof VOICE_CHANGER_CLIENT_EXCEPTION[keyof typeof VOICE_CHANGER_CLIENT_EXCEPTION]


////////////////////////////////////
// indexedDB
////////////////////////////////////
export const INDEXEDDB_DB_APP_NAME = "INDEXEDDB_KEY_VOICE_CHANGER"
export const INDEXEDDB_DB_NAME = "INDEXEDDB_KEY_VOICE_CHANGER_DB"
export const INDEXEDDB_KEY_CLIENT = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_CLIENT"
export const INDEXEDDB_KEY_SERVER = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_SERVER"
export const INDEXEDDB_KEY_WORKLETNODE = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_WORKLETNODE"
export const INDEXEDDB_KEY_MODEL_DATA = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_MODEL_DATA"
export const INDEXEDDB_KEY_WORKLET = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_WORKLET"


