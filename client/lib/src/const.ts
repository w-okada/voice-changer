
// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。
// 24000sample -> 1sec, 128sample(1chunk) -> 5.333msec
// 187.5chunk -> 1sec

export const ClientType = {
    "MMVCv15": "MMVCv15",
    "MMVCv13": "MMVCv13",
    "so-vits-svc-40": "so-vits-svc-40",
    "so-vits-svc-40_c": "so-vits-svc-40_c",
    "so-vits-svc-40v2": "so-vits-svc-40v2",
    "DDSP-SVC": "DDSP-SVC",
    "RVC": "RVC"

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

export const ModelSamplingRate = {
    "48000": 48000,
    "40000": 40000,
    "32000": 32000
} as const
export type ModelSamplingRate = typeof InputSampleRate[keyof typeof InputSampleRate]


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
    // "parselmouth": "parselmouth",
    "crepe": "crepe",
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

    "enableServerAudio": "enableServerAudio",
    "serverAudioStated": "serverAudioStated",
    "serverInputAudioSampleRate": "serverInputAudioSampleRate",
    "serverOutputAudioSampleRate": "serverOutputAudioSampleRate",
    "serverInputAudioBufferSize": "serverInputAudioBufferSize",
    "serverOutputAudioBufferSize": "serverOutputAudioBufferSize",
    "serverInputDeviceId": "serverInputDeviceId",
    "serverOutputDeviceId": "serverOutputDeviceId",
    "serverReadChunkSize": "serverReadChunkSize",

    "tran": "tran",
    "noiseScale": "noiseScale",
    "predictF0": "predictF0",
    "silentThreshold": "silentThreshold",
    "extraConvertSize": "extraConvertSize",
    "clusterInferRatio": "clusterInferRatio",

    "indexRatio": "indexRatio",
    "rvcQuality": "rvcQuality",
    "modelSamplingRate": "modelSamplingRate",
    "silenceFront": "silenceFront",
    "modelSlotIndex": "modelSlotIndex",

    "useEnhancer": "useEnhancer",
    "useDiff": "useDiff",
    "useDiffDpm": "useDiffDpm",
    "useDiffSilence": "useDiffSilence",
    "diffAcc": "diffAcc",
    "diffSpkId": "diffSpkId",
    "kStep": "kStep",
    "threshold": "threshold",

    "inputSampleRate": "inputSampleRate",
    "enableDirectML": "enableDirectML",
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

    enableServerAudio: number // 0:off, 1:on
    serverAudioStated: number // 0:off, 1:on
    serverInputAudioSampleRate: number
    serverOutputAudioSampleRate: number
    serverInputAudioBufferSize: number
    serverOutputAudioBufferSize: number
    serverInputDeviceId: number
    serverOutputDeviceId: number
    serverReadChunkSize: number


    tran: number // so-vits-svc
    noiseScale: number // so-vits-svc
    predictF0: number // so-vits-svc
    silentThreshold: number // so-vits-svc
    extraConvertSize: number// so-vits-svc
    clusterInferRatio: number // so-vits-svc

    indexRatio: number // RVC
    rvcQuality: number // 0:low, 1:high
    silenceFront: number // 0:off, 1:on
    modelSamplingRate: ModelSamplingRate // 32000,40000,48000
    modelSlotIndex: number,

    useEnhancer: number// DDSP-SVC
    useDiff: number// DDSP-SVC
    useDiffDpm: number// DDSP-SVC
    useDiffSilence: number// DDSP-SVC
    diffAcc: number// DDSP-SVC
    diffSpkId: number// DDSP-SVC
    kStep: number// DDSP-SVC
    threshold: number// DDSP-SVC

    inputSampleRate: InputSampleRate
    enableDirectML: number
}

type ModelSlot = {
    modelFile: string
    featureFile: string,
    indexFile: string,

    defaultTrans: number,

    modelType: number,
    embChannels: number,
    f0: boolean,
    samplingRate: number
    deprecated: boolean
}

type ServerAudioDevice = {
    kind: "audioinput" | "audiooutput",
    index: number,
    name: string
    hostAPI: string
}

export type ServerInfo = VoiceChangerServerSetting & {
    status: string
    configFile: string,
    pyTorchModelFile: string,
    onnxModelFile: string,
    onnxExecutionProviders: OnnxExecutionProvider[]
    modelSlots: ModelSlot[]
    serverAudioInputDevices: ServerAudioDevice[]
    serverAudioOutputDevices: ServerAudioDevice[]

}

export type ServerInfoSoVitsSVC = ServerInfo & {
    speakers: { [key: string]: number }
}

export const DefaultServerSetting: ServerInfo = {
    // VC Common 
    inputSampleRate: 48000,

    crossFadeOffsetRate: 0.0,
    crossFadeEndRate: 1.0,
    crossFadeOverlapSize: CrossFadeOverlapSize[1024],

    recordIO: 0,

    enableServerAudio: 0,
    serverAudioStated: 0,
    serverInputAudioSampleRate: 48000,
    serverOutputAudioSampleRate: 48000,
    serverInputAudioBufferSize: 1024 * 24,
    serverOutputAudioBufferSize: 1024 * 24,
    serverInputDeviceId: -1,
    serverOutputDeviceId: -1,
    serverReadChunkSize: 256,

    // VC Specific
    srcId: 0,
    dstId: 1,
    gpu: 0,


    framework: Framework.PyTorch,
    f0Factor: 1.0,
    onnxExecutionProvider: OnnxExecutionProvider.CPUExecutionProvider,
    f0Detector: F0Detector.dio,

    tran: 0,
    noiseScale: 0,
    predictF0: 0,
    silentThreshold: 0,
    extraConvertSize: 0,
    clusterInferRatio: 0,

    indexRatio: 0,
    rvcQuality: 0,
    modelSamplingRate: 48000,
    silenceFront: 1,
    modelSlotIndex: 0,

    useEnhancer: 0,
    useDiff: 1,
    useDiffDpm: 0,
    useDiffSilence: 0,
    diffAcc: 20,
    diffSpkId: 1,
    kStep: 120,
    threshold: -45,

    enableDirectML: 0,
    // 
    status: "ok",
    configFile: "",
    pyTorchModelFile: "",
    onnxModelFile: "",
    onnxExecutionProviders: [],
    modelSlots: [],
    serverAudioInputDevices: [],
    serverAudioOutputDevices: []
}
export const DefaultServerSetting_MMVCv15: ServerInfo = {
    ...DefaultServerSetting, dstId: 101,
}
export const DefaultServerSetting_MMVCv13: ServerInfo = {
    ...DefaultServerSetting, srcId: 107, dstId: 100,
}

export const DefaultServerSetting_so_vits_svc_40: ServerInfo = {
    ...DefaultServerSetting, tran: 10, noiseScale: 0.3, extraConvertSize: 1024 * 32, clusterInferRatio: 0.1,
}

export const DefaultServerSetting_so_vits_svc_40_c: ServerInfo = {
    ...DefaultServerSetting, tran: 10, noiseScale: 0.3, extraConvertSize: 1024 * 32, clusterInferRatio: 0.1,
}
export const DefaultServerSetting_so_vits_svc_40v2: ServerInfo = {
    ...DefaultServerSetting, tran: 10, noiseScale: 0.3, extraConvertSize: 1024 * 32, clusterInferRatio: 0.1,
}

export const DefaultServerSetting_DDSP_SVC: ServerInfo = {
    ...DefaultServerSetting, dstId: 1, tran: 10, extraConvertSize: 1024 * 32
}


export const DefaultServerSetting_RVC: ServerInfo = {
    ...DefaultServerSetting, tran: 10, extraConvertSize: 1024 * 32, f0Detector: F0Detector.harvest
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
    sendingSampleRate: 48000,
    inputChunkNum: 48,
    downSamplingMode: "average"
}

export const DefaultWorkletNodeSetting_so_vits_svc_40: WorkletNodeSetting = {
    ...DefaultWorkletNodeSetting, inputChunkNum: 128,
}

export const DefaultWorkletNodeSetting_so_vits_svc_40v2: WorkletNodeSetting = {
    ...DefaultWorkletNodeSetting, inputChunkNum: 128,
}

export const DefaultWorkletNodeSetting_DDSP_SVC: WorkletNodeSetting = {
    ...DefaultWorkletNodeSetting, inputChunkNum: 256,
}

export const DefaultWorkletNodeSetting_RVC: WorkletNodeSetting = {
    ...DefaultWorkletNodeSetting, inputChunkNum: 256,
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


// ONNX
export type OnnxExporterInfo = {
    "status": string
    "path": string
    "filename": string
}


export const MAX_MODEL_SLOT_NUM = 3

// Merge
export type MergeElement = {
    filename: string
    strength: number
}
export type MergeModelRequest = {
    command: "mix",
    defaultTrans: number,
    files: MergeElement[]
}
