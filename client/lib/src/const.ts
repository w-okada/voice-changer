
// (★1) chunk sizeは 128サンプル, 256byte(int16)と定義。
// (★2) 256byte(最低バッファサイズ256から間引いた個数x2byte)をchunkとして管理。


// types
export type VoiceChangerRequestParamas = {
    convertChunkNum: number,
    srcId: number,
    dstId: number,
    gpu: number,

    crossFadeLowerValue: number,
    crossFadeOffsetRate: number,
    crossFadeEndRate: number,
}

export type VoiceChangerRequest = VoiceChangerRequestParamas & {
    data: ArrayBuffer,
    timestamp: number
}

export type VoiceChangerOptions = {
    audioInputDeviceId: string | null,
    mediaStream: MediaStream | null,
    mmvcServerUrl: string,
    sampleRate: SampleRate, // 48000Hz
    bufferSize: BufferSize, // 256, 512, 1024, 2048, 4096, 8192, 16384 (for mic stream)
    chunkNum: number, // n of (256 x n) for send buffer
    speakers: Speaker[],
    forceVfDisable: boolean,
    voiceChangerMode: VoiceChangerMode,
}


export type Speaker = {
    "id": number,
    "name": string,
}

// Consts

export const MajarModeTypes = {
    "sio": "sio",
    "rest": "rest",
} as const
export type MajarModeTypes = typeof MajarModeTypes[keyof typeof MajarModeTypes]

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
    "1024": 1024,
} as const
export type BufferSize = typeof BufferSize[keyof typeof BufferSize]


// Defaults
export const DefaultVoiceChangerRequestParamas: VoiceChangerRequestParamas = {
    convertChunkNum: 12, //（★１）
    srcId: 107,
    dstId: 100,
    gpu: 0,
    crossFadeLowerValue: 0.1,
    crossFadeOffsetRate: 0.3,
    crossFadeEndRate: 0.6
}

export const DefaultSpeakders: Speaker[] = [
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
]

