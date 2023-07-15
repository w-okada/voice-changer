
// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。
// 24000sample -> 1sec, 128sample(1chunk) -> 5.333msec
// 187.5chunk -> 1sec

export const VoiceChangerType = {
    "MMVCv15": "MMVCv15",
    "MMVCv13": "MMVCv13",
    "so-vits-svc-40": "so-vits-svc-40",
    "DDSP-SVC": "DDSP-SVC",
    "RVC": "RVC",
    "Diffusion-SVC":"Diffusion-SVC"

} as const
export type VoiceChangerType = typeof VoiceChangerType[keyof typeof VoiceChangerType]

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

export const F0Detector = {
    "dio": "dio",
    "harvest": "harvest",
    "crepe": "crepe",
    "crepe_full": "crepe_full",
    "crepe_tiny": "crepe_tiny",
} as const
export type F0Detector = typeof F0Detector[keyof typeof F0Detector]

export const DiffMethod = {
    "pndm": "pndm",
    "dpm-solver": "dpm-solver",
} as const
export type DiffMethod = typeof DiffMethod[keyof typeof DiffMethod]

export const RVCModelType = {
    "pyTorchRVC": "pyTorchRVC",
    "pyTorchRVCNono": "pyTorchRVCNono",
    "pyTorchRVCv2": "pyTorchRVCv2",
    "pyTorchRVCv2Nono": "pyTorchRVCv2Nono",
    "pyTorchWebUI": "pyTorchWebUI",
    "pyTorchWebUINono": "pyTorchWebUINono",
    "onnxRVC": "onnxRVC",
    "onnxRVCNono": "onnxRVCNono",
} as const
export type RVCModelType = typeof RVCModelType[keyof typeof RVCModelType]

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
    "serverAudioSampleRate": "serverAudioSampleRate",
    "serverInputAudioSampleRate": "serverInputAudioSampleRate",
    "serverOutputAudioSampleRate": "serverOutputAudioSampleRate",
    "serverMonitorAudioSampleRate": "serverMonitorAudioSampleRate",
    "serverInputAudioBufferSize": "serverInputAudioBufferSize",
    "serverOutputAudioBufferSize": "serverOutputAudioBufferSize",
    "serverInputDeviceId": "serverInputDeviceId",
    "serverOutputDeviceId": "serverOutputDeviceId",
    "serverMonitorDeviceId": "serverMonitorDeviceId",
    "serverReadChunkSize": "serverReadChunkSize",
    "serverInputAudioGain": "serverInputAudioGain",
    "serverOutputAudioGain": "serverOutputAudioGain",

    "tran": "tran",
    "noiseScale": "noiseScale",
    "predictF0": "predictF0",
    "silentThreshold": "silentThreshold",
    "extraConvertSize": "extraConvertSize",
    "clusterInferRatio": "clusterInferRatio",

    "indexRatio": "indexRatio",
    "protect": "protect",
    "rvcQuality": "rvcQuality",
    "modelSamplingRate": "modelSamplingRate",
    "silenceFront": "silenceFront",
    "modelSlotIndex": "modelSlotIndex",

    "useEnhancer": "useEnhancer",
    "useDiff": "useDiff",
    // "useDiffDpm": "useDiffDpm",
    "diffMethod": "diffMethod",
    "useDiffSilence": "useDiffSilence",
    "diffAcc": "diffAcc",
    "diffSpkId": "diffSpkId",
    "kStep": "kStep",
    "threshold": "threshold",

    "speedUp": "speedUp",

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

    f0Factor: number
    f0Detector: F0Detector // dio or harvest
    recordIO: number // 0:off, 1:on

    enableServerAudio: number // 0:off, 1:on
    serverAudioStated: number // 0:off, 1:on
    serverAudioSampleRate: number
    serverInputAudioSampleRate: number
    serverOutputAudioSampleRate: number
    serverMonitorAudioSampleRate: number
    serverInputAudioBufferSize: number
    serverOutputAudioBufferSize: number
    serverInputDeviceId: number
    serverOutputDeviceId: number
    serverMonitorDeviceId: number
    serverReadChunkSize: number
    serverInputAudioGain: number
    serverOutputAudioGain: number


    tran: number // so-vits-svc
    noiseScale: number // so-vits-svc
    predictF0: number // so-vits-svc
    silentThreshold: number // so-vits-svc
    extraConvertSize: number// so-vits-svc
    clusterInferRatio: number // so-vits-svc

    indexRatio: number // RVC
    protect: number // RVC
    rvcQuality: number // 0:low, 1:high
    silenceFront: number // 0:off, 1:on
    modelSamplingRate: ModelSamplingRate // 32000,40000,48000
    modelSlotIndex: number,

    useEnhancer: number// DDSP-SVC
    useDiff: number// DDSP-SVC
    // useDiffDpm: number// DDSP-SVC
    diffMethod: DiffMethod, // DDSP-SVC
    useDiffSilence: number// DDSP-SVC
    diffAcc: number// DDSP-SVC
    diffSpkId: number// DDSP-SVC
    kStep: number// DDSP-SVC
    threshold: number// DDSP-SVC

    speedUp: number // Diffusion-SVC


    inputSampleRate: InputSampleRate
    enableDirectML: number
}

type ModelSlot = {
    voiceChangerType: VoiceChangerType
    name: string,
    description: string,
    credit: string,
    termsOfUseUrl: string,
    iconFile: string
    speakers: { [key: number]: string }
}

export type RVCModelSlot = ModelSlot & {
    modelFile: string
    indexFile: string,
    defaultIndexRatio: number,
    defaultProtect: number,
    defaultTune: number,
    modelType: RVCModelType,

    embChannels: number,
    f0: boolean,
    samplingRate: number
    deprecated: boolean
}

export type MMVCv13ModelSlot = ModelSlot & {
    modelFile: string
    configFile: string,
    srcId: number
    dstId: number

    samplingRate: number
    speakers: { [key: number]: string }
}

export type MMVCv15ModelSlot = ModelSlot & {
    modelFile: string
    configFile: string,
    srcId: number
    dstId: number
    f0Factor: number
    samplingRate: number
    f0: { [key: number]: number }
}

export type SoVitsSvc40ModelSlot = ModelSlot & {
    modelFile: string
    configFile: string,
    clusterFile: string,
    dstId: number

    samplingRate: number

    defaultTune: number
    defaultClusterInferRatio: number
    noiseScale: number
    speakers: { [key: number]: string }
}

export type DDSPSVCModelSlot = ModelSlot & {
    modelFile: string
    configFile: string,
    diffModelFile: string
    diffConfigFile: string
    dstId: number

    samplingRate: number

    defaultTune: number
    enhancer: boolean
    diffusion: boolean
    acc: number
    kstep: number
    speakers: { [key: number]: string }
}


export type DiffusionSVCModelSlot = ModelSlot & {
    modelFile: string
    dstId: number

    samplingRate: number

    defaultTune: number
    defaultKstep : number
    defaultSpeedup: number
    kStepMax: number
    speakers: { [key: number]: string }
}

export type ModelSlotUnion = RVCModelSlot | MMVCv13ModelSlot | MMVCv15ModelSlot | SoVitsSvc40ModelSlot | DDSPSVCModelSlot | DiffusionSVCModelSlot

type ServerAudioDevice = {
    kind: "audioinput" | "audiooutput",
    index: number,
    name: string
    hostAPI: string
}

export type ServerInfo = VoiceChangerServerSetting & {
    // コンフィグ対象外 (getInfoで取得のみ可能な情報)
    status: string
    modelSlots: ModelSlotUnion[]
    serverAudioInputDevices: ServerAudioDevice[]
    serverAudioOutputDevices: ServerAudioDevice[]
    sampleModels: RVCSampleModel[]
    gpus: {
        id: number,
        name: string,
        memory: number,
    }[]
    maxInputLength: number  // MMVCv15

}


export type RVCSampleModel = {
    id: string
    name: string
    modelUrl: string
    indexUrl: string
    featureUrl: string
    termsOfUseUrl: string
    credit: string
    description: string
    lang: string
    tag: string[]
    icon: string
    f0: boolean
    sampleRate: number
    modelType: string
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
    serverAudioSampleRate: 48000,
    serverInputAudioSampleRate: 48000,
    serverOutputAudioSampleRate: 48000,
    serverMonitorAudioSampleRate: 48000,
    serverInputAudioBufferSize: 1024 * 24,
    serverOutputAudioBufferSize: 1024 * 24,
    serverInputDeviceId: -1,
    serverOutputDeviceId: -1,
    serverMonitorDeviceId: -1,
    serverReadChunkSize: 256,
    serverInputAudioGain: 1.0,
    serverOutputAudioGain: 1.0,

    // VC Specific
    srcId: 0,
    dstId: 1,
    gpu: 0,


    f0Factor: 1.0,
    f0Detector: F0Detector.dio,

    tran: 0,
    noiseScale: 0,
    predictF0: 0,
    silentThreshold: 0,
    extraConvertSize: 0,
    clusterInferRatio: 0,

    indexRatio: 0,
    protect: 0.5,
    rvcQuality: 0,
    modelSamplingRate: 48000,
    silenceFront: 1,
    modelSlotIndex: 0,
    sampleModels: [],
    gpus: [],

    useEnhancer: 0,
    useDiff: 1,
    diffMethod: "dpm-solver",
    useDiffSilence: 0,
    diffAcc: 20,
    diffSpkId: 1,
    kStep: 120,
    threshold: -45,

    speedUp: 10,

    enableDirectML: 0,
    // 
    status: "ok",
    modelSlots: [],
    serverAudioInputDevices: [],
    serverAudioOutputDevices: [],

    maxInputLength:  128 * 2048
}

///////////////////////
// Workletセッティング
///////////////////////

export type WorkletSetting = {
    numTrancateTreshold: number,
    volTrancateThreshold: number,
    volTrancateLength: number
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


///////////////////////
// クライアントセッティング
///////////////////////
export const SampleRate = {
    "48000": 48000,
} as const
export type SampleRate = typeof SampleRate[keyof typeof SampleRate]

export type VoiceChangerClientSetting = {
    audioInput: string | MediaStream | null,
    sampleRate: SampleRate, // 48000Hz
    echoCancel: boolean,
    noiseSuppression: boolean,
    noiseSuppression2: boolean

    inputGain: number
    outputGain: number
}

///////////////////////
// Client セッティング
///////////////////////
export type ClientSetting = {
    workletSetting: WorkletSetting
    workletNodeSetting: WorkletNodeSetting
    voiceChangerClientSetting: VoiceChangerClientSetting
}
export const DefaultClientSettng: ClientSetting = {
    workletSetting: {
        numTrancateTreshold: 100,
        volTrancateThreshold: 0.0005,
        volTrancateLength: 32
    },
    workletNodeSetting: {
        serverUrl: "",
        protocol: "sio",
        sendingSampleRate: 48000,
        inputChunkNum: 48,
        downSamplingMode: "average"
    },
    voiceChangerClientSetting: {
        audioInput: null,
        sampleRate: 48000,
        echoCancel: false,
        noiseSuppression: false,
        noiseSuppression2: false,
        inputGain: 1.0,
        outputGain: 1.0
    }
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
export const INDEXEDDB_KEY_MODEL_DATA = "INDEXEDDB_KEY_VOICE_CHANGER_LIB_MODEL_DATA"


// ONNX
export type OnnxExporterInfo = {
    "status": string
    "path": string
    "filename": string
}

// Merge
export type MergeElement = {
    filename: string
    strength: number
}
export type MergeModelRequest = {
    voiceChangerType: VoiceChangerType
    command: "mix",
    files: MergeElement[]
}
