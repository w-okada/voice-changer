import { VoiceChangerWorkletProcessorRequest } from "./@types/voice-changer-worklet-processor";
import { DefaultWorkletNodeSetting, DownSamplingMode, VOICE_CHANGER_CLIENT_EXCEPTION, WorkletNodeSetting, WorkletSetting } from "./const";
import { io, Socket } from "socket.io-client";
import { DefaultEventsMap } from "@socket.io/component-emitter";

export type VoiceChangerWorkletListener = {
    notifyVolume: (vol: number) => void
    notifySendBufferingTime: (time: number) => void
    notifyResponseTime: (time: number, perf?: number[]) => void
    notifyException: (code: VOICE_CHANGER_CLIENT_EXCEPTION, message: string) => void
}

export class VoiceChangerWorkletNode extends AudioWorkletNode {
    private listener: VoiceChangerWorkletListener

    private setting: WorkletNodeSetting = DefaultWorkletNodeSetting
    private requestChunks: ArrayBuffer[] = []
    private socket: Socket<DefaultEventsMap, DefaultEventsMap> | null = null
    // performance monitor
    private bufferStart = 0;

    private isOutputRecording = false;
    private recordingOutputChunk: Float32Array[] = []
    private outputNode: VoiceChangerWorkletNode | null = null

    constructor(context: AudioContext, listener: VoiceChangerWorkletListener) {
        super(context, "voice-changer-worklet-processor");
        this.port.onmessage = this.handleMessage.bind(this);
        this.listener = listener
        this.createSocketIO()
        console.log(`[worklet_node][voice-changer-worklet-processor] created.`);
    }

    setOutputNode = (outputNode: VoiceChangerWorkletNode | null) => {
        this.outputNode = outputNode
    }


    // 設定
    updateSetting = (setting: WorkletNodeSetting) => {
        console.log(`[WorkletNode] Updating WorkletNode Setting,`, this.setting, setting)
        let recreateSocketIoRequired = false
        if (this.setting.serverUrl != setting.serverUrl || this.setting.protocol != setting.protocol) {
            recreateSocketIoRequired = true
        }
        this.setting = setting
        if (recreateSocketIoRequired) {
            this.createSocketIO()
        }
    }

    getSettings = (): WorkletNodeSetting => {
        return this.setting
    }

    getSocketId = () => {
        return this.socket?.id
    }

    // 処理
    private createSocketIO = () => {
        if (this.socket) {
            this.socket.close()
        }
        if (this.setting.protocol === "sio") {
            this.socket = io(this.setting.serverUrl + "/test");
            this.socket.on('connect_error', (err) => {
                this.listener.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_SIO_CONNECT_FAILED, `[SIO] rconnection failed ${err}`)
            })
            this.socket.on('connect', () => {
                console.log(`[SIO] sonnect to ${this.setting.serverUrl}`)
                console.log(`[SIO] ${this.socket?.id}`)
            });
            this.socket.on('response', (response: any[]) => {
                const cur = Date.now()
                const responseTime = cur - response[0]
                const result = response[1] as ArrayBuffer
                const perf = response[2]
                if (result.byteLength < 128 * 2) {
                    this.listener.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_SIO_INVALID_RESPONSE, `[SIO] recevied data is too short ${result.byteLength}`)
                } else {
                    if (this.outputNode != null) {
                        this.outputNode.postReceivedVoice(response[1])
                    } else {
                        this.postReceivedVoice(response[1])
                    }
                    this.listener.notifyResponseTime(responseTime, perf)
                }
            });
        }
    }

    postReceivedVoice = (data: ArrayBuffer) => {
        // Int16 to Float
        const i16Data = new Int16Array(data)
        const f32Data = new Float32Array(i16Data.length)
        // console.log(`[worklet] f32DataLength${f32Data.length} i16DataLength${i16Data.length}`)
        i16Data.forEach((x, i) => {
            const float = (x >= 0x8000) ? -(0x10000 - x) / 0x8000 : x / 0x7FFF;
            f32Data[i] = float
        })

        // アップサンプリング
        let upSampledBuffer: Float32Array | null = null
        if (this.setting.sendingSampleRate == 48000) {
            upSampledBuffer = f32Data
        } else {
            upSampledBuffer = new Float32Array(f32Data.length * 2)
            for (let i = 0; i < f32Data.length; i++) {
                const currentFrame = f32Data[i]
                const nextFrame = i + 1 < f32Data.length ? f32Data[i + 1] : f32Data[i]
                upSampledBuffer[i * 2] = currentFrame
                upSampledBuffer[i * 2 + 1] = (currentFrame + nextFrame) / 2
            }
        }

        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "voice",
            voice: upSampledBuffer,
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)

        if (this.isOutputRecording) {
            this.recordingOutputChunk.push(upSampledBuffer)
        }

    }

    private _averageDownsampleBuffer(buffer: Float32Array, originalSampleRate: number, destinationSamplerate: number) {
        if (originalSampleRate == destinationSamplerate) {
            return buffer;
        }
        if (destinationSamplerate > originalSampleRate) {
            throw "downsampling rate show be smaller than original sample rate";
        }
        const sampleRateRatio = originalSampleRate / destinationSamplerate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < result.length) {
            var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            // Use average value of skipped samples
            var accum = 0, count = 0;
            for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count++;
            }
            result[offsetResult] = accum / count;
            // Or you can simply get rid of the skipped samples:
            // result[offsetResult] = buffer[nextOffsetBuffer];
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    handleMessage(event: any) {
        // console.log(`[Node:handleMessage_] `, event.data.volume);
        if (event.data.responseType === "volume") {
            this.listener.notifyVolume(event.data.volume as number)
        } else if (event.data.responseType === "inputData") {
            const inputData = event.data.inputData as Float32Array
            // console.log("receive input data", inputData)

            // ダウンサンプリング
            let downsampledBuffer: Float32Array | null = null
            if (this.setting.sendingSampleRate == 48000) {
                downsampledBuffer = inputData
            } else if (this.setting.downSamplingMode == DownSamplingMode.decimate) {
                //////// (Kind 1) 間引き //////////
                //// 48000Hz で入ってくるので間引いて24000Hzに変換する。
                downsampledBuffer = new Float32Array(inputData.length / 2);
                for (let i = 0; i < inputData.length; i++) {
                    if (i % 2 == 0) {
                        downsampledBuffer[i / 2] = inputData[i]
                    }
                }
            } else {
                //////// (Kind 2) 平均 //////////
                // downsampledBuffer = this._averageDownsampleBuffer(buffer, 48000, 24000)
                downsampledBuffer = this._averageDownsampleBuffer(inputData, 48000, this.setting.sendingSampleRate)
            }

            // Float to Int16
            const arrayBuffer = new ArrayBuffer(downsampledBuffer.length * 2)
            const dataView = new DataView(arrayBuffer);
            for (let i = 0; i < downsampledBuffer.length; i++) {
                let s = Math.max(-1, Math.min(1, downsampledBuffer[i]));
                s = s < 0 ? s * 0x8000 : s * 0x7FFF
                dataView.setInt16(i * 2, s, true);
            }

            // バッファリング
            this.requestChunks.push(arrayBuffer)

            //// リクエストバッファの中身が、リクエスト送信数と違う場合は処理終了。
            if (this.requestChunks.length < this.setting.inputChunkNum) {
                return
            }

            // リクエスト用の入れ物を作成
            const windowByteLength = this.requestChunks.reduce((prev, cur) => {
                return prev + cur.byteLength
            }, 0)
            const newBuffer = new Uint8Array(windowByteLength);

            // リクエストのデータをセット
            this.requestChunks.reduce((prev, cur) => {
                newBuffer.set(new Uint8Array(cur), prev)
                return prev + cur.byteLength
            }, 0)


            this.sendBuffer(newBuffer)
            this.requestChunks = []

            this.listener.notifySendBufferingTime(Date.now() - this.bufferStart)
            this.bufferStart = Date.now()

        } else {
            console.warn(`[worklet_node][voice-changer-worklet-processor] unknown response ${event.data.responseType}`, event.data)
        }
    }



    private sendBuffer = async (newBuffer: Uint8Array) => {
        const timestamp = Date.now()
        if (this.setting.protocol === "sio") {
            if (!this.socket) {
                console.warn(`sio is not initialized`)
                return
            }
            // console.log("emit!")
            this.socket.emit('request_message', [
                timestamp,
                newBuffer.buffer]);
        } else {
            const res = await postVoice(
                this.setting.serverUrl + "/test",
                timestamp,
                newBuffer.buffer)

            if (res.byteLength < 128 * 2) {
                this.listener.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_REST_INVALID_RESPONSE, `[REST] recevied data is too short ${res.byteLength}`)
            } else {
                if (this.outputNode != null) {
                    this.outputNode.postReceivedVoice(res)
                } else {
                    this.postReceivedVoice(res)
                }
                this.listener.notifyResponseTime(Date.now() - timestamp)
            }
        }
    }


    configure = (setting: WorkletSetting) => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "config",
            voice: new Float32Array(1),
            numTrancateTreshold: setting.numTrancateTreshold,
            volTrancateThreshold: setting.volTrancateThreshold,
            volTrancateLength: setting.volTrancateLength
        }
        this.port.postMessage(req)
    }

    start = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "start",
            voice: new Float32Array(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)

    }
    stop = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "stop",
            voice: new Float32Array(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)
    }
    trancateBuffer = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "trancateBuffer",
            voice: new Float32Array(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)
    }

    startOutputRecording = () => {
        this.recordingOutputChunk = []
        this.isOutputRecording = true
    }
    stopOutputRecording = () => {
        this.isOutputRecording = false

        const dataSize = this.recordingOutputChunk.reduce((prev, cur) => {
            return prev + cur.length
        }, 0)
        const samples = new Float32Array(dataSize);
        let sampleIndex = 0
        for (let i = 0; i < this.recordingOutputChunk.length; i++) {
            for (let j = 0; j < this.recordingOutputChunk[i].length; j++) {
                samples[sampleIndex] = this.recordingOutputChunk[i][j];
                sampleIndex++;
            }
        }
        return samples
    }
}


export const postVoice = async (
    url: string,
    timestamp: number,
    buffer: ArrayBuffer) => {
    const obj = {
        timestamp,
        buffer: Buffer.from(buffer).toString('base64')
    };
    const body = JSON.stringify(obj);

    const res = await fetch(`${url}`, {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: body
    })

    const receivedJson = await res.json()
    const changedVoiceBase64 = receivedJson["changedVoiceBase64"]
    const buf = Buffer.from(changedVoiceBase64, "base64")
    const ab = new ArrayBuffer(buf.length);
    const view = new Uint8Array(ab);
    for (let i = 0; i < buf.length; ++i) {
        view[i] = buf[i];
    }
    return ab
}