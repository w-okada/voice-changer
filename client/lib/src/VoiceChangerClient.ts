import { VoiceChangerWorkletNode } from "./VoiceChangerWorkletNode";
// @ts-ignore
import workerjs from "raw-loader!../worklet/dist/index.js";
import { VoiceFocusDeviceTransformer, VoiceFocusTransformDevice } from "amazon-chime-sdk-js";
import { createDummyMediaStream } from "./util";
import { BufferSize, MajarModeTypes, VoiceChangerMode, VoiceChangerRequestParamas } from "./const";
import MicrophoneStream from "microphone-stream";
import { AudioStreamer, Callbacks, AudioStreamerListeners } from "./AudioStreamer";


// オーディオデータの流れ
// input node(mic or MediaStream) -> [vf node] -> microphne stream -> audio streamer -> 
//    sio/rest server -> audio streamer-> vc node -> output node



export class VoiceChnagerClient {
    private ctx: AudioContext
    private vfEnable = false
    private vf: VoiceFocusDeviceTransformer | null = null
    private currentDevice: VoiceFocusTransformDevice | null = null

    private currentMediaStream: MediaStream | null = null
    private currentMediaStreamAudioSourceNode: MediaStreamAudioSourceNode | null = null
    private outputNodeFromVF: MediaStreamAudioDestinationNode | null = null
    private micStream: MicrophoneStream | null = null
    private audioStreamer!: AudioStreamer
    private vcNode!: VoiceChangerWorkletNode
    private currentMediaStreamAudioDestinationNode!: MediaStreamAudioDestinationNode

    private promiseForInitialize: Promise<void>

    private callbacks: Callbacks = {
        onVoiceReceived: (voiceChangerMode: VoiceChangerMode, data: ArrayBuffer): void => {
            console.log(voiceChangerMode, data)
            if (voiceChangerMode === "realtime") {
                this.vcNode.postReceivedVoice(data)
                return
            }

            // For Near Realtime Mode
            console.log("near realtime mode")

            const i16Data = new Int16Array(data)
            const f32Data = new Float32Array(i16Data.length)
            // https://stackoverflow.com/questions/35234551/javascript-converting-from-int16-to-float32
            i16Data.forEach((x, i) => {
                const float = (x >= 0x8000) ? -(0x10000 - x) / 0x8000 : x / 0x7FFF;
                f32Data[i] = float

            })

            const source = this.ctx.createBufferSource();
            const buffer = this.ctx.createBuffer(1, f32Data.length, 24000);
            buffer.getChannelData(0).set(f32Data);
            source.buffer = buffer;
            source.start();
            source.connect(this.currentMediaStreamAudioDestinationNode)
        }
    }

    constructor(ctx: AudioContext, vfEnable: boolean, audioStreamerListeners: AudioStreamerListeners) {
        this.ctx = ctx
        this.vfEnable = vfEnable
        this.promiseForInitialize = new Promise<void>(async (resolve) => {
            const scriptUrl = URL.createObjectURL(new Blob([workerjs], { type: "text/javascript" }));
            await this.ctx.audioWorklet.addModule(scriptUrl)

            this.vcNode = new VoiceChangerWorkletNode(this.ctx); // vc node 
            this.currentMediaStreamAudioDestinationNode = this.ctx.createMediaStreamDestination() // output node
            this.vcNode.connect(this.currentMediaStreamAudioDestinationNode) // vc node -> output node
            // (vc nodeにはaudio streamerのcallbackでデータが投げ込まれる)
            this.audioStreamer = new AudioStreamer("sio", this.callbacks, audioStreamerListeners, { objectMode: true, })


            if (this.vfEnable) {
                this.vf = await VoiceFocusDeviceTransformer.create({ variant: 'c20' })
                const dummyMediaStream = createDummyMediaStream(this.ctx)
                this.currentDevice = (await this.vf.createTransformDevice(dummyMediaStream)) || null;
                this.outputNodeFromVF = this.ctx.createMediaStreamDestination();
            }
            resolve()
        })
    }

    isInitialized = async () => {
        if (this.promiseForInitialize) {
            await this.promiseForInitialize
        }
        return true
    }

    // forceVfDisable is for the condition that vf is enabled in constructor. 
    setup = async (input: string | MediaStream, bufferSize: BufferSize, forceVfDisable: boolean = false) => {
        // condition check
        if (!this.vcNode) {
            console.warn("vc node is not initialized.")
            throw "vc node is not initialized."
        }

        // Main Process
        //// shutdown & re-generate mediastream
        if (this.currentMediaStream) {
            this.currentMediaStream.getTracks().forEach(x => { x.stop() })
            this.currentMediaStream = null
        }
        if (typeof input == "string") {
            this.currentMediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { deviceId: input }
            })
        } else {
            this.currentMediaStream = input
        }

        // create mic stream
        this.micStream = new MicrophoneStream({
            objectMode: true,
            bufferSize: bufferSize,
            context: this.ctx
        })

        // connect nodes.
        if (this.currentDevice && forceVfDisable == false) {
            this.currentMediaStreamAudioSourceNode = this.ctx.createMediaStreamSource(this.currentMediaStream) // input node
            this.currentDevice.chooseNewInnerDevice(this.currentMediaStream)
            const voiceFocusNode = await this.currentDevice.createAudioNode(this.ctx); // vf node
            this.currentMediaStreamAudioSourceNode.connect(voiceFocusNode.start) // input node -> vf node
            voiceFocusNode.end.connect(this.outputNodeFromVF!)
            this.micStream.setStream(this.outputNodeFromVF!.stream) // vf node -> mic stream
        } else {
            this.micStream.setStream(this.currentMediaStream) // input device -> mic stream
        }
        this.micStream.pipe(this.audioStreamer!) // mic stream -> audio streamer


    }
    get stream(): MediaStream {
        return this.currentMediaStreamAudioDestinationNode.stream
    }



    // Audio Streamer Settingg
    setServerUrl = (serverUrl: string, mode: MajarModeTypes) => {
        this.audioStreamer.setServerUrl(serverUrl, mode)
    }

    setRequestParams = (val: VoiceChangerRequestParamas) => {
        this.audioStreamer.setRequestParams(val)
    }

    setChunkNum = (num: number) => {
        this.audioStreamer.setChunkNum(num)
    }

    setVoiceChangerMode = (val: VoiceChangerMode) => {
        this.audioStreamer.setVoiceChangerMode(val)
    }
}