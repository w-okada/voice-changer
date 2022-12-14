import { io, Socket } from "socket.io-client";
import { DefaultEventsMap } from "@socket.io/component-emitter";
import { Duplex, DuplexOptions } from "readable-stream";
import { Protocol, VoiceChangerMode, VOICE_CHANGER_CLIENT_EXCEPTION } from "./const";

export type Callbacks = {
    onVoiceReceived: (voiceChangerMode: VoiceChangerMode, data: ArrayBuffer) => void
}
export type AudioStreamerListeners = {
    notifySendBufferingTime: (time: number) => void
    notifyResponseTime: (time: number) => void
    notifyException: (code: VOICE_CHANGER_CLIENT_EXCEPTION, message: string) => void
}

export type AudioStreamerSettings = {
    serverUrl: string;
    protocol: Protocol;
    inputChunkNum: number;
    voiceChangerMode: VoiceChangerMode;
}

export class AudioStreamer extends Duplex {
    private callbacks: Callbacks
    private audioStreamerListeners: AudioStreamerListeners
    private protocol: Protocol = "sio"
    private serverUrl = ""
    private socket: Socket<DefaultEventsMap, DefaultEventsMap> | null = null
    private voiceChangerMode: VoiceChangerMode = "realtime"
    private inputChunkNum = 128
    private requestChunks: ArrayBuffer[] = []
    private recordChunks: ArrayBuffer[] = []
    private isRecording = false

    // performance monitor
    private bufferStart = 0;

    constructor(callbacks: Callbacks, audioStreamerListeners: AudioStreamerListeners, options?: DuplexOptions) {
        super(options);
        this.callbacks = callbacks
        this.audioStreamerListeners = audioStreamerListeners
    }

    private createSocketIO = () => {
        if (this.socket) {
            this.socket.close()
        }
        if (this.protocol === "sio") {
            this.socket = io(this.serverUrl + "/test");
            this.socket.on('connect_error', (err) => {
                this.audioStreamerListeners.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_SIO_CONNECT_FAILED, `[SIO] rconnection failed ${err}`)
            })
            this.socket.on('connect', () => console.log(`[SIO] sonnect to ${this.serverUrl}`));
            this.socket.on('response', (response: any[]) => {
                const cur = Date.now()
                const responseTime = cur - response[0]
                const result = response[1] as ArrayBuffer
                if (result.byteLength < 128 * 2) {
                    this.audioStreamerListeners.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_SIO_INVALID_RESPONSE, `[SIO] recevied data is too short ${result.byteLength}`)
                } else {
                    this.callbacks.onVoiceReceived(this.voiceChangerMode, response[1])
                    this.audioStreamerListeners.notifyResponseTime(responseTime)
                }
            });
        }
    }

    // Option Change
    setServerUrl = (serverUrl: string) => {
        this.serverUrl = serverUrl
        console.log(`[AudioStreamer] Server Setting:${this.serverUrl} ${this.protocol}`)
        this.createSocketIO()// mode check is done in the method.
    }
    setProtocol = (mode: Protocol) => {
        this.protocol = mode
        console.log(`[AudioStreamer] Server Setting:${this.serverUrl} ${this.protocol}`)
        this.createSocketIO()// mode check is done in the method.
    }

    setInputChunkNum = (num: number) => {
        this.inputChunkNum = num
    }

    setVoiceChangerMode = (val: VoiceChangerMode) => {
        this.voiceChangerMode = val
    }

    getSettings = (): AudioStreamerSettings => {
        return {
            serverUrl: this.serverUrl,
            protocol: this.protocol,
            inputChunkNum: this.inputChunkNum,
            voiceChangerMode: this.voiceChangerMode
        }
    }


    // Main Process
    //// Pipe from mic stream 
    _write = (chunk: AudioBuffer, _encoding: any, callback: any) => {
        const buffer = chunk.getChannelData(0);
        // console.log("SAMPLERATE:", chunk.sampleRate, chunk.numberOfChannels, chunk.length, buffer)
        if (this.voiceChangerMode === "realtime") {
            this._write_realtime(buffer)
        } else {
            this._write_record(buffer)
        }
        callback();
    }

    private _write_realtime = (buffer: Float32Array) => {
        // bufferSize??????????????????48Khz????????????????????????
        //// 48000Hz ????????????????????????????????????24000Hz??????????????????
        //// ???????????????????????????????????????(x1/2), 16bit(2byte)???(x2)
        const arrayBuffer = new ArrayBuffer((buffer.length / 2) * 2)
        const dataView = new DataView(arrayBuffer);

        for (let i = 0; i < buffer.length; i++) {
            if (i % 2 == 0) {
                let s = Math.max(-1, Math.min(1, buffer[i]));
                s = s < 0 ? s * 0x8000 : s * 0x7FFF
                // ???????????????????????????????????????????????????((i/2)*2)
                dataView.setInt16((i / 2) * 2, s, true);
            }
        }
        // 256byte(???????????????????????????256????????????????????????x2byte)???chunk???????????????
        const chunkByteSize = 256 // (const.ts ???1)
        for (let i = 0; i < arrayBuffer.byteLength / chunkByteSize; i++) {
            const ab = arrayBuffer.slice(i * chunkByteSize, (i + 1) * chunkByteSize)
            this.requestChunks.push(ab)
        }

        //// ???????????????????????????????????????????????????????????????????????????????????????????????????
        if (this.requestChunks.length < this.inputChunkNum) {
            return
        }

        // ???????????????????????????????????????
        const windowByteLength = this.requestChunks.reduce((prev, cur) => {
            return prev + cur.byteLength
        }, 0)
        const newBuffer = new Uint8Array(windowByteLength);

        // ???????????????????????????????????????
        this.requestChunks.reduce((prev, cur) => {
            newBuffer.set(new Uint8Array(cur), prev)
            return prev + cur.byteLength
        }, 0)

        // console.log("send buff length", newBuffer.length)

        this.sendBuffer(newBuffer)
        this.requestChunks = []

        this.audioStreamerListeners.notifySendBufferingTime(Date.now() - this.bufferStart)
        this.bufferStart = Date.now()
    }


    private _write_record = (buffer: Float32Array) => {
        if (!this.isRecording) { return }
        // buffer(for48Khz)x16bit * chunksize / 2(for24Khz)
        const sendBuffer = new ArrayBuffer(buffer.length * 2 / 2);
        const sendDataView = new DataView(sendBuffer);
        for (var i = 0; i < buffer.length; i++) {
            if (i % 2 == 0) {
                let s = Math.max(-1, Math.min(1, buffer[i]));
                s = s < 0 ? s * 0x8000 : s * 0x7FFF
                sendDataView.setInt16(i, s, true);
                // if (i % 3000 === 0) {
                //     console.log("buffer_converting", s, buffer[i])
                // }
            }
        }
        this.recordChunks.push(sendBuffer)
    }

    // Near Realtime???????????????
    sendRecordedData = () => {
        const length = this.recordChunks.reduce((prev, cur) => {
            return prev + cur.byteLength
        }, 0)
        const newBuffer = new Uint8Array(length);
        this.recordChunks.reduce((prev, cur) => {
            newBuffer.set(new Uint8Array(cur), prev)
            return prev + cur.byteLength
        }, 0)

        this.sendBuffer(newBuffer)
    }

    startRecord = () => {
        this.recordChunks = []
        this.isRecording = true
    }

    stopRecord = () => {
        this.isRecording = false
    }

    private sendBuffer = async (newBuffer: Uint8Array) => {
        // if (this.serverUrl.length == 0) {
        //     // console.warn("no server url")
        //     // return
        //     // throw "no server url"
        // }
        const timestamp = Date.now()
        // console.log("REQUEST_MESSAGE:", [this.gpu, this.srcId, this.dstId, timestamp, newBuffer.buffer])
        // console.log("SERVER_URL", this.serverUrl, this.protocol)
        // const convertChunkNum = this.voiceChangerMode === "realtime" ? this.requestParamas.convertChunkNum : 0
        if (this.protocol === "sio") {
            if (!this.socket) {
                console.warn(`sio is not initialized`)
                return
            }
            // console.log("emit!")
            this.socket.emit('request_message', [
                // this.requestParamas.gpu,
                // this.requestParamas.srcId,
                // this.requestParamas.dstId,
                timestamp,
                // convertChunkNum,
                // this.requestParamas.crossFadeLowerValue,
                // this.requestParamas.crossFadeOffsetRate,
                // this.requestParamas.crossFadeEndRate,
                newBuffer.buffer]);
        } else {
            const res = await postVoice(
                this.serverUrl + "/test",
                // this.requestParamas.gpu,
                // this.requestParamas.srcId,
                // this.requestParamas.dstId,
                timestamp,
                // convertChunkNum,
                // this.requestParamas.crossFadeLowerValue,
                // this.requestParamas.crossFadeOffsetRate,
                // this.requestParamas.crossFadeEndRate,
                newBuffer.buffer)

            if (res.byteLength < 128 * 2) {
                this.audioStreamerListeners.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_REST_INVALID_RESPONSE, `[REST] recevied data is too short ${res.byteLength}`)
            } else {
                this.callbacks.onVoiceReceived(this.voiceChangerMode, res)
                this.audioStreamerListeners.notifyResponseTime(Date.now() - timestamp)
            }
        }
    }
}

export const postVoice = async (
    url: string,
    // gpu: number,
    // srcId: number,
    // dstId: number,
    timestamp: number,
    // convertChunkNum: number,
    // crossFadeLowerValue: number,
    // crossFadeOffsetRate: number,
    // crossFadeEndRate: number,
    buffer: ArrayBuffer) => {
    const obj = {
        // gpu,
        // srcId,
        // dstId,
        timestamp,
        // convertChunkNum,
        // crossFadeLowerValue,
        // crossFadeOffsetRate,
        // crossFadeEndRate,
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
    // console.log("RECIV", buf.length)
    const view = new Uint8Array(ab);
    for (let i = 0; i < buf.length; ++i) {
        view[i] = buf[i];
    }
    return ab
}