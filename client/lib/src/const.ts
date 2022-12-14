
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
}

export type VoiceChangerClientSetting = {
    audioInput: string | MediaStream | null,
    mmvcServerUrl: string,
    protocol: Protocol,
    sampleRate: SampleRate, // 48000Hz
    bufferSize: BufferSize, // 256, 512, 1024, 2048, 4096, 8192, 16384 (for mic stream)
    inputChunkNum: number, // n of (256 x n) for send buffer
    speakers: Speaker[],
    forceVfDisable: boolean,
    voiceChangerMode: VoiceChangerMode,
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
    "onnxExecutionProvider": "onnxExecutionProvider"
} as const
export type ServerSettingKey = typeof ServerSettingKey[keyof typeof ServerSettingKey]

// Defaults
export const DefaultVoiceChangerServerSetting: VoiceChangerServerSetting = {
    convertChunkNum: 32, //（★１）
    minConvertSize: 0,
    srcId: 107,
    dstId: 100,
    gpu: 0,
    crossFadeLowerValue: 0.1,
    crossFadeOffsetRate: 0.1,
    crossFadeEndRate: 0.9,
    crossFadeOverlapRate: 0.5,
    framework: "PyTorch",
    onnxExecutionProvider: "CPUExecutionProvider"

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
            "id": 100,
            "name": "ずんだもん"
        },
        {
            "id": 107,
            "name": "user"
        },
        {
            "id": 101,
            "name": "そら"
        },
        {
            "id": 102,
            "name": "めたん"
        },
        {
            "id": 103,
            "name": "つむぎ"
        }
    ],
    forceVfDisable: false,
    voiceChangerMode: "realtime",
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


