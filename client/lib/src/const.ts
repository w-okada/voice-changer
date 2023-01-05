
// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。


// types
export type VoiceChangerRequestParamas = {
    convertChunkNum: number, // VITSに入力する変換サイズ。(入力データの2倍以上の大きさで指定。それより小さいものが指定された場合は、サーバ側で自動的に入力の2倍のサイズが設定される。)
    srcId: number,
    dstId: number,
    gpu: number,

    crossFadeLowerValue: number,
    crossFadeOffsetRate: number,
    crossFadeEndRate: number,
}

export type VoiceChangerOptions = {
    audioInputDeviceId: string | null,
    mediaStream: MediaStream | null,
    mmvcServerUrl: string,
    protocol: Protocol,
    sampleRate: SampleRate, // 48000Hz
    bufferSize: BufferSize, // 256, 512, 1024, 2048, 4096, 8192, 16384 (for mic stream)
    inputChunkNum: number, // n of (256 x n) for send buffer
    speakers: Speaker[],
    forceVfDisable: boolean,
    voiceChangerMode: VoiceChangerMode,
}


export type Speaker = {
    "id": number,
    "name": string,
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


// Defaults
export const DefaultVoiceChangerRequestParamas: VoiceChangerRequestParamas = {
    convertChunkNum: 1, //（★１）
    srcId: 107,
    dstId: 100,
    gpu: 0,
    crossFadeLowerValue: 0.1,
    crossFadeOffsetRate: 0.1,
    crossFadeEndRate: 0.9
}

export const DefaultVoiceChangerOptions: VoiceChangerOptions = {
    audioInputDeviceId: null,
    mediaStream: null,
    mmvcServerUrl: "https://192.168.0.3:18888/test",
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
    voiceChangerMode: "realtime"
}


export const VOICE_CHANGER_CLIENT_EXCEPTION = {
    ERR_SIO_CONNECT_FAILED: "ERR_SIO_CONNECT_FAILED",
    ERR_SIO_INVALID_RESPONSE: "ERR_SIO_INVALID_RESPONSE",
    ERR_REST_INVALID_RESPONSE: "ERR_REST_INVALID_RESPONSE"

} as const
export type VOICE_CHANGER_CLIENT_EXCEPTION = typeof VOICE_CHANGER_CLIENT_EXCEPTION[keyof typeof VOICE_CHANGER_CLIENT_EXCEPTION]


