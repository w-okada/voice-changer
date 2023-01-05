
class VoiceChangerWorkletProcessor extends AudioWorkletProcessor {
    private BLOCK_SIZE = 128
    private initialized = false;
    private volume = 0
    playBuffer: Float32Array[] = []
    /**
     * @constructor
     */
    constructor() {
        super();
        this.initialized = true;
        this.port.onmessage = this.handleMessage.bind(this);
    }

    handleMessage(event: any) {
        // noop
        const arrayBuffer = event.data.data as ArrayBuffer
        // データは(int16)で受信
        const i16Data = new Int16Array(arrayBuffer)
        const f32Data = new Float32Array(i16Data.length)
        // console.log(`[worklet] f32DataLength${f32Data.length} i16DataLength${i16Data.length}`)
        i16Data.forEach((x, i) => {
            const float = (x >= 0x8000) ? -(0x10000 - x) / 0x8000 : x / 0x7FFF;
            f32Data[i] = float
        })

        if (this.playBuffer.length > 50) {
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

        const data = this.playBuffer.shift()!

        const sum = data.reduce((prev, cur) => {
            return prev + cur * cur
        }, 0)
        const rms = Math.sqrt(sum / data.length)

        this.volume = Math.max(rms, this.volume * 0.95)
        this.port.postMessage({ volume: this.volume });



        outputs[0][0].set(data)

        return true;
    }
}
registerProcessor("voice-changer-worklet-processor", VoiceChangerWorkletProcessor);
