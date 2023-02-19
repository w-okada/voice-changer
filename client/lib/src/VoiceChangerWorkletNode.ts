import { VoiceChangerWorkletProcessorRequest } from "./@types/voice-changer-worklet-processor";
import { WorkletSetting } from "./const";
import { io, Socket } from "socket.io-client";
import { DefaultEventsMap } from "@socket.io/component-emitter";

export type VoiceChangerWorkletListener = {
    notifyVolume: (vol: number) => void
    notifyOutputRecordData: (data: Float32Array[]) => void
}

export class VoiceChangerWorkletNode extends AudioWorkletNode {
    private listener: VoiceChangerWorkletListener
    private requestChunks: ArrayBuffer[] = []
    private socket: Socket<DefaultEventsMap, DefaultEventsMap> | null = null
    constructor(context: AudioContext, listener: VoiceChangerWorkletListener) {
        super(context, "voice-changer-worklet-processor");
        this.port.onmessage = this.handleMessage.bind(this);
        this.listener = listener
        this.createSocketIO()
        console.log(`[worklet_node][voice-changer-worklet-processor] created.`);
    }

    postReceivedVoice = (data: ArrayBuffer) => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "voice",
            voice: data,
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)
    }


    private createSocketIO = () => {
        if (this.socket) {
            this.socket.close()
        }
        // if (this.setting.protocol === "sio") {
        // this.socket = io(this.setting.serverUrl + "/test");
        this.socket = io("/test");
        this.socket.on('connect_error', (err) => {
            console.log("connect exception !!!!!")
            // this.audioStreamerListeners.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_SIO_CONNECT_FAILED, `[SIO] rconnection failed ${err}`)
        })
        this.socket.on('connect', () => {
            // console.log(`[SIO] sonnect to ${this.setting.serverUrl}`)
            console.log(`[SIO] ${this.socket?.id}`)
        });
        this.socket.on('response', (response: any[]) => {
            const cur = Date.now()
            const responseTime = cur - response[0]
            const result = response[1] as ArrayBuffer
            if (result.byteLength < 128 * 2) {
                console.log("tooshort!!")
                // this.audioStreamerListeners.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_SIO_INVALID_RESPONSE, `[SIO] recevied data is too short ${result.byteLength}`)
            } else {
                console.log("response!!!")
                this.postReceivedVoice(response[1])
                // this.callbacks.onVoiceReceived(response[1])
                // this.audioStreamerListeners.notifyResponseTime(responseTime)
            }
        });
        // }
    }

    handleMessage(event: any) {
        // console.log(`[Node:handleMessage_] `, event.data.volume);
        if (event.data.responseType === "volume") {
            this.listener.notifyVolume(event.data.volume as number)
        } else if (event.data.responseType === "recordData") {
            this.listener.notifyOutputRecordData(event.data.recordData as Float32Array[])
        } else if (event.data.responseType === "inputData") {
            const inputData = event.data.inputData as Float32Array
            // console.log("receive input data", inputData)

            const arrayBuffer = new ArrayBuffer(inputData.length * 2)
            const dataView = new DataView(arrayBuffer);
            for (let i = 0; i < inputData.length; i++) {
                let s = Math.max(-1, Math.min(1, inputData[i]));
                s = s < 0 ? s * 0x8000 : s * 0x7FFF
                dataView.setInt16(i * 2, s, true);
            }

            this.requestChunks.push(arrayBuffer)

            //// リクエストバッファの中身が、リクエスト送信数と違う場合は処理終了。
            if (this.requestChunks.length < 32) {
                return
            }
            console.log("sending...")

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
            console.log("sended...")
            this.requestChunks = []



        } else {
            console.warn(`[worklet_node][voice-changer-worklet-processor] unknown response ${event.data.responseType}`, event.data)
        }
    }



    private sendBuffer = async (newBuffer: Uint8Array) => {
        const timestamp = Date.now()
        // if (this.setting.protocol === "sio") {
        if (!this.socket) {
            console.warn(`sio is not initialized`)
            return
        }
        // console.log("emit!")
        this.socket.emit('request_message', [
            timestamp,
            newBuffer.buffer]);
        // } else {
        //     const res = await postVoice(
        //         this.setting.serverUrl + "/test",
        //         timestamp,
        //         newBuffer.buffer)

        //     if (res.byteLength < 128 * 2) {
        //         this.audioStreamerListeners.notifyException(VOICE_CHANGER_CLIENT_EXCEPTION.ERR_REST_INVALID_RESPONSE, `[REST] recevied data is too short ${res.byteLength}`)
        //     } else {
        //         this.callbacks.onVoiceReceived(res)
        //         this.audioStreamerListeners.notifyResponseTime(Date.now() - timestamp)
        //     }
        // }
    }


    configure = (setting: WorkletSetting) => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "config",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: setting.numTrancateTreshold,
            volTrancateThreshold: setting.volTrancateThreshold,
            volTrancateLength: setting.volTrancateLength
        }
        this.port.postMessage(req)
    }

    startOutputRecordingWorklet = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "startRecording",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)

    }
    stopOutputRecordingWorklet = () => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "stopRecording",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: 0,
            volTrancateThreshold: 0,
            volTrancateLength: 0
        }
        this.port.postMessage(req)
    }
}