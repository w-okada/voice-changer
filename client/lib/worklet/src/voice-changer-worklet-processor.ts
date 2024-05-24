export const RequestType = {
    voice: "voice",
    config: "config",
    start: "start",
    stop: "stop",
    trancateBuffer: "trancateBuffer",
} as const;
export type RequestType = (typeof RequestType)[keyof typeof RequestType];

export const ResponseType = {
    volume: "volume",
    inputData: "inputData",
    start_ok: "start_ok",
    stop_ok: "stop_ok",
} as const;
export type ResponseType = (typeof ResponseType)[keyof typeof ResponseType];

export type VoiceChangerWorkletProcessorRequest = {
    requestType: RequestType;
    voice: Float32Array;
    numTrancateTreshold: number;
    volTrancateThreshold: number;
    volTrancateLength: number;
};

export type VoiceChangerWorkletProcessorResponse = {
    responseType: ResponseType;
    volume?: number;
    recordData?: Float32Array[];
    inputData?: Float32Array;
};

class VoiceChangerWorkletProcessor extends AudioWorkletProcessor {
    private BLOCK_SIZE = 128;
    private initialized = false;
    private volume = 0;
    // private numTrancateTreshold = 100;
    // private volTrancateThreshold = 0.0005
    // private volTrancateLength = 32
    // private volTrancateCount = 0

    private isRecording = false;

    playBuffer: Float32Array[] = [];
    /**
     * @constructor
     */
    constructor() {
        super();
        console.log("[AudioWorkletProcessor] created.");
        this.initialized = true;
        this.port.onmessage = this.handleMessage.bind(this);
    }

    calcVol = (data: Float32Array, prevVol: number) => {
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i]
        }
        const rms = Math.sqrt((sum * sum) / data.length);
        return Math.max(rms, prevVol * 0.95);
    };

    trancateBuffer = (start: number, end?: number) => {
        this.playBuffer = this.playBuffer.slice(start, end)
        console.log("[worklet] Buffer truncated");
    };
    handleMessage(event: any) {
        const request = event.data as VoiceChangerWorkletProcessorRequest;
        if (request.requestType === "config") {
            // this.numTrancateTreshold = request.numTrancateTreshold;
            // this.volTrancateLength = request.volTrancateLength
            // this.volTrancateThreshold = request.volTrancateThreshold
            console.log("[worklet] worklet configured", request);
            return;
        } else if (request.requestType === "start") {
            if (this.isRecording) {
                console.warn("[worklet] recoring is already started");
                return;
            }
            this.isRecording = true;
            const startResponse: VoiceChangerWorkletProcessorResponse = {
                responseType: "start_ok",
            };
            this.port.postMessage(startResponse);
            return;
        } else if (request.requestType === "stop") {
            if (!this.isRecording) {
                console.warn("[worklet] recoring is not started");
                return;
            }
            this.isRecording = false;
            const stopResponse: VoiceChangerWorkletProcessorResponse = {
                responseType: "stop_ok",
            };
            this.port.postMessage(stopResponse);
            return;
        } else if (request.requestType === "trancateBuffer") {
            this.trancateBuffer(0, 0);
            return;
        }

        const f32Data = request.voice;
        const chunkSize = Math.floor(f32Data.length / this.BLOCK_SIZE);
        const bufferCutoff = chunkSize * 1.25;
        if (this.playBuffer.length > bufferCutoff) {
            console.log(`[worklet] Truncate ${this.playBuffer.length} > ${bufferCutoff}`);
            this.trancateBuffer(this.playBuffer.length - bufferCutoff);
        }

        for (let i = 0; i < chunkSize; i++) {
            const block = f32Data.subarray(i * this.BLOCK_SIZE, (i + 1) * this.BLOCK_SIZE);
            this.playBuffer.push(block);
        }
    }

    pushData = (inputData: Float32Array) => {
        const volumeResponse: VoiceChangerWorkletProcessorResponse = {
            responseType: ResponseType.inputData,
            inputData: inputData,
        };
        this.port.postMessage(volumeResponse, [inputData.buffer]);
    };

    process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>) {
        if (!this.initialized) {
            console.warn("[worklet] worklet_process not ready");
            return true;
        }

        if (this.isRecording) {
            if (_inputs.length > 0 && _inputs[0].length > 0) {
                this.pushData(_inputs[0][0]);
            }
        }

        // console.log("[worklet] play buffer");
        //// 一定期間無音状態が続いている場合はスキップ。
        // let voice: Float32Array | undefined
        // while (true) {
        //     voice = this.playBuffer.shift()
        //     if (!voice) {
        //         break
        //     }
        //     this.volume = this.calcVol(voice, this.volume)
        //     if (this.volume < this.volTrancateThreshold) {
        //         this.volTrancateCount += 1
        //     } else {
        //         this.volTrancateCount = 0
        //     }

        //     // V.1.5.0よりsilent skipで音飛びするようになったので無効化
        //     if (this.volTrancateCount < this.volTrancateLength || this.volTrancateLength < 0) {
        //         break
        //     } else {
        //         break
        //         // console.log("silent...skip")
        //     }
        // }
        const voice = this.playBuffer.shift();
        if (voice) {
            this.volume = this.calcVol(voice, this.volume);
            const volumeResponse: VoiceChangerWorkletProcessorResponse = {
                responseType: ResponseType.volume,
                volume: this.volume,
            };
            this.port.postMessage(volumeResponse);
            outputs[0][0].set(voice);
            if (outputs[0].length == 2) {
                outputs[0][1].set(voice);
            }
        }

        return true;
    }
}
registerProcessor("voice-changer-worklet-processor", VoiceChangerWorkletProcessor);
