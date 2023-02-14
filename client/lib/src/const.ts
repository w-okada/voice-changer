
// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。
// 24000sample -> 1sec, 128sample(1chunk) -> 5.333msec
// 187.5chunk -> 1sec


// types
export type VoiceChangerServerSetting = {
    convertChunkNum: number, // VITSに入力する変換サイズ。(入力データの2倍以上の大きさで指定。それより小さいものが指定された場合は、サーバ側で自動的に入力の2倍のサイズが設定される。)
    minConvertSize: number, // この値より小さい場合にこの値に揃える。

    srcId: number,
    dstId: number,
    gpu: number,

    crossFadeLowerValue: number,
    crossFadeOffsetRate: number,
    crossFadeEndRate: number,
    crossFadeOverlapRate: number,

    framework: Framework
    onnxExecutionProvider: OnnxExecutionProvider,

    f0Factor: number
    f0Detector: string // dio or harvest
    recordIO: number // 0:off, 1:on
}

export type VoiceChangerClientSetting = {
    audioInput: string | MediaStream | null,
    mmvcServerUrl: string,
    protocol: Protocol,
    sampleRate: SampleRate, // 48000Hz
    bufferSize: BufferSize, // 256, 512, 1024, 2048, 4096, 8192, 16384 (for mic stream)
    inputChunkNum: number, // n of (256 x n) for send buffer
    speakers: Speaker[],
    correspondences: Correspondence[],
    echoCancel: boolean,
    noiseSuppression: boolean,
    noiseSuppression2: boolean,
    voiceChangerMode: VoiceChangerMode,
    downSamplingMode: DownSamplingMode,

    inputGain: number
    outputGain: number
}

export type WorkletSetting = {
    numTrancateTreshold: number,
    volTrancateThreshold: number,
    volTrancateLength: number
}

export type Speaker = {
    "id": number,
    "name": string,
}
export type Correspondence = {
    "sid": number,
    "correspondence": number,
    "dirname": string
}


export type ServerInfo = {
    status: string
    configFile: string,
    pyTorchModelFile: string,
    onnxModelFile: string,
    convertChunkNum: number,
    minConvertSize: number,
    crossFadeOffsetRate: number,
    crossFadeEndRate: number,
    crossFadeOverlapRate: number,
    gpu: number,
    srcId: number,
    dstId: number,
    framework: Framework,
    onnxExecutionProvider: string[]
    f0Factor: number
    f0Detector: string
    recordIO: number
}





// Consts
export const Protocol = {
    "sio": "sio",
    "rest": "rest",
} as const
export type Protocol = typeof Protocol[keyof typeof Protocol]

export const VoiceChangerMode = {
    "realtime": "realtime",
    "near-realtime": "near-realtime",
} as const
export type VoiceChangerMode = typeof VoiceChangerMode[keyof typeof VoiceChangerMode]

export const DownSamplingMode = {
    "decimate": "decimate",
    "average": "average"
} as const
export type DownSamplingMode = typeof DownSamplingMode[keyof typeof DownSamplingMode]

export const SampleRate = {
    "48000": 48000,
} as const
export type SampleRate = typeof SampleRate[keyof typeof SampleRate]

export const BufferSize = {
    "256": 256,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
    "4096": 4096,
    "8192": 8192,
    "16384": 16384
} as const
export type BufferSize = typeof BufferSize[keyof typeof BufferSize]

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
}
export type Framework = typeof Framework[keyof typeof Framework]

export const F0Detector = {
    "dio": "dio",
    "harvest": "harvest",
}
export type F0Detector = typeof F0Detector[keyof typeof F0Detector]

export const ServerSettingKey = {
    "srcId": "srcId",
    "dstId": "dstId",
    "convertChunkNum": "convertChunkNum",
    "minConvertSize": "minConvertSize",
    "gpu": "gpu",
    "crossFadeOffsetRate": "crossFadeOffsetRate",
    "crossFadeEndRate": "crossFadeEndRate",
    "crossFadeOverlapRate": "crossFadeOverlapRate",
    "framework": "framework",
    "onnxExecutionProvider": "onnxExecutionProvider",
    "f0Factor": "f0Factor",
    "f0Detector": "f0Detector",
    "recordIO": "recordIO"
} as const
export type ServerSettingKey = typeof ServerSettingKey[keyof typeof ServerSettingKey]

// Defaults
export const DefaultVoiceChangerServerSetting: VoiceChangerServerSetting = {
    convertChunkNum: 32, //（★１）
    minConvertSize: 0,
    srcId: 0,
    dstId: 101,
    gpu: 0,
    crossFadeLowerValue: 0.1,
    crossFadeOffsetRate: 0.1,
    crossFadeEndRate: 0.9,
    crossFadeOverlapRate: 0.5,
    framework: "PyTorch",
    f0Factor: 1.0,
    onnxExecutionProvider: "CPUExecutionProvider",
    f0Detector: "dio",
    recordIO: 0
}

export const DefaultVoiceChangerClientSetting: VoiceChangerClientSetting = {
    audioInput: null,
    mmvcServerUrl: "",
    protocol: "sio",
    sampleRate: 48000,
    bufferSize: 1024,
    inputChunkNum: 48,
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
    echoCancel: true,
    noiseSuppression: true,
    noiseSuppression2: false,
    voiceChangerMode: "realtime",
    downSamplingMode: "average",
    inputGain: 1.0,
    outputGain: 1.0
}

export const DefaultWorkletSetting: WorkletSetting = {
    numTrancateTreshold: 188,
    volTrancateThreshold: 0.0005,
    volTrancateLength: 32
}

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
export const INDEXEDDB_KEY_MODEL_DATA = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_MODEL_DATA"
export const INDEXEDDB_KEY_WORKLET = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_WORKLET"


