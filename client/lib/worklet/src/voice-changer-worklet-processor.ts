export const RequestType = {
    "voice": "voice",
    "config": "config"
} as const
export type RequestType = typeof RequestType[keyof typeof RequestType]

export type VoiceChangerWorkletProcessorRequest = {
    requestType: RequestType,
    voice: ArrayBuffer,
    numTrancateTreshold: number
    volTrancateThreshold: number
    volTrancateLength: number
}

class VoiceChangerWorkletProcessor extends AudioWorkletProcessor {
    private BLOCK_SIZE = 128
    private initialized = false;
    private volume = 0
    private numTrancateTreshold = 50
    private volTrancateThreshold = 0.0005
    private volTrancateLength = 32
    private volTrancateCount = 0

    playBuffer: Float32Array[] = []
    /**
     * @constructor
     */
    constructor() {
        super();
        this.initialized = true;
        this.port.onmessage = this.handleMessage.bind(this);
    }

    calcVol = (data: Float32Array, prevVol: number) => {
        const sum = data.reduce((prev, cur) => {
            return prev + cur * cur
        }, 0)
        const rms = Math.sqrt(sum / data.length)
        return Math.max(rms, prevVol * 0.95)
    }

    handleMessage(event: any) {
        const request = event.data.request as VoiceChangerWorkletProcessorRequest
        if (request.requestType === "config") {
            this.numTrancateTreshold = request.numTrancateTreshold
            this.volTrancateLength = request.volTrancateLength
            this.volTrancateThreshold = request.volTrancateThreshold
            console.log("[worklet] worklet configured", request)
            return
        }

        const arrayBuffer = request.voice
        // データは(int16)で受信
        const i16Data = new Int16Array(arrayBuffer)
        const f32Data = new Float32Array(i16Data.length)
        // console.log(`[worklet] f32DataLength${f32Data.length} i16DataLength${i16Data.length}`)
        i16Data.forEach((x, i) => {
            const float = (x >= 0x8000) ? -(0x10000 - x) / 0x8000 : x / 0x7FFF;
            f32Data[i] = float
        })

        if (this.playBuffer.length > this.numTrancateTreshold) {
            console.log("[worklet] Buffer truncated")
            while (this.playBuffer.length > 2) {
                this.playBuffer.shift()
            }
        }

        // アップサンプリングしてPlayバッファに蓄積
        let f32Block: Float32Array
        for (let i = 0; i < f32Data.length; i++) {
            const frameIndexInBlock = (i * 2) % this.BLOCK_SIZE //
            if (frameIndexInBlock === 0) {
                f32Block = new Float32Array(this.BLOCK_SIZE)
            }

            const currentFrame = f32Data[i]
            const nextFrame = i + 1 < f32Data.length ? f32Data[i + 1] : f32Data[i]
            f32Block![frameIndexInBlock] = currentFrame
            f32Block![frameIndexInBlock + 1] = (currentFrame + nextFrame) / 2
            if (f32Block!.length === frameIndexInBlock + 2) {
                this.playBuffer.push(f32Block!)
            }
        }
    }


    process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>) {
        if (!this.initialized) {
            console.warn("[worklet] worklet_process not ready");
            return true;
        }

        if (this.playBuffer.length === 0) {
            console.log("[worklet] no play buffer")
            return true
        }

        //// 一定期間無音状態が続いている場合はスキップ。
        let voice: Float32Array | undefined
        while (true) {
            voice = this.playBuffer.shift()
            if (!voice) {
                break
            }
            this.volume = this.calcVol(voice, this.volume)
            if (this.volume < this.volTrancateThreshold) {
                this.volTrancateCount += 1
            } else {
                this.volTrancateCount = 0
            }


            if (this.volTrancateCount < this.volTrancateLength) {
                break
            } else {
                console.log("silent...skip")
            }
        }

        if (voice) {
            this.port.postMessage({ volume: this.volume });
            outputs[0][0].set(voice)
        }

        return true;
    }
}
registerProcessor("voice-changer-worklet-processor", VoiceChangerWorkletProcessor);
