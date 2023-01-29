import { VoiceChangerWorkletNode, VolumeListener } from "./VoiceChangerWorkletNode";
// @ts-ignore
import workerjs from "raw-loader!../worklet/dist/index.js";
import { VoiceFocusDeviceTransformer, VoiceFocusTransformDevice } from "amazon-chime-sdk-js";
import { createDummyMediaStream, validateUrl } from "./util";
import { BufferSize, DefaultVoiceChangerClientSetting, Protocol, ServerSettingKey, VoiceChangerMode, VOICE_CHANGER_CLIENT_EXCEPTION, WorkletSetting } from "./const";
import MicrophoneStream from "microphone-stream";
import { AudioStreamer, Callbacks, AudioStreamerListeners } from "./AudioStreamer";
import { ServerConfigurator } from "./ServerConfigurator";
import { VoiceChangerWorkletProcessorRequest } from "./@types/voice-changer-worklet-processor";

// オーディオデータの流れ
// input node(mic or MediaStream) -> [vf node] -> microphne stream -> audio streamer -> 
//    sio/rest server -> audio streamer-> vc node -> output node

import { BlockingQueue } from "./utils/BlockingQueue";

export class VoiceChangerClient {
    private configurator: ServerConfigurator
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
    private _isVoiceChanging = false

    private sslCertified: string[] = []

    private sem = new BlockingQueue<number>();

    private callbacks: Callbacks = {
        onVoiceReceived: (voiceChangerMode: VoiceChangerMode, data: ArrayBuffer): void => {
            // console.log(voiceChangerMode, data)
            if (voiceChangerMode === "realtime") {
                const req: VoiceChangerWorkletProcessorRequest = {
                    requestType: "voice",
                    voice: data,
                    numTrancateTreshold: 0,
                    volTrancateThreshold: 0,
                    volTrancateLength: 0
                }

                this.vcNode.postReceivedVoice(req)
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

    constructor(ctx: AudioContext, vfEnable: boolean, audioStreamerListeners: AudioStreamerListeners, volumeListener: VolumeListener) {
        this.sem.enqueue(0);
        this.configurator = new ServerConfigurator()
        this.ctx = ctx
        this.vfEnable = vfEnable
        this.promiseForInitialize = new Promise<void>(async (resolve) => {
            const scriptUrl = URL.createObjectURL(new Blob([workerjs], { type: "text/javascript" }));
            await this.ctx.audioWorklet.addModule(scriptUrl)

            this.vcNode = new VoiceChangerWorkletNode(this.ctx, volumeListener); // vc node 
            this.currentMediaStreamAudioDestinationNode = this.ctx.createMediaStreamDestination() // output node
            this.vcNode.connect(this.currentMediaStreamAudioDestinationNode) // vc node -> output node
            // (vc nodeにはaudio streamerのcallbackでデータが投げ込まれる)
            this.audioStreamer = new AudioStreamer(this.callbacks, audioStreamerListeners, { objectMode: true, })
            this.audioStreamer.setInputChunkNum(DefaultVoiceChangerClientSetting.inputChunkNum)
            this.audioStreamer.setVoiceChangerMode(DefaultVoiceChangerClientSetting.voiceChangerMode)

            if (this.vfEnable) {
                this.vf = await VoiceFocusDeviceTransformer.create({ variant: 'c20' })
                const dummyMediaStream = createDummyMediaStream(this.ctx)
                this.currentDevice = (await this.vf.createTransformDevice(dummyMediaStream)) || null;
                this.outputNodeFromVF = this.ctx.createMediaStreamDestination();
            }
            resolve()
        })
    }

    private lock = async () => {
        const num = await this.sem.dequeue();
        return num;
    };
    private unlock = (num: number) => {
        this.sem.enqueue(num + 1);
    };


    isInitialized = async () => {
        if (this.promiseForInitialize) {
            await this.promiseForInitialize
        }
        return true
    }

    // forceVfDisable is for the condition that vf is enabled in constructor. 
    setup = async (input: string | MediaStream, bufferSize: BufferSize, forceVfDisable: boolean = false) => {
        const lockNum = await this.lock()
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
        if (this.micStream) {
            this.micStream.unpipe()
            this.micStream.destroy()
            this.micStream = null
        }
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
            console.log("VF disabled")
            this.micStream.setStream(this.currentMediaStream) // input device -> mic stream
        }
        this.micStream.pipe(this.audioStreamer) // mic stream -> audio streamer
        if (!this._isVoiceChanging) {
            this.micStream.pauseRecording()
        } else {
            this.micStream.playRecording()
        }
        await this.unlock(lockNum)
    }
    get stream(): MediaStream {
        return this.currentMediaStreamAudioDestinationNode.stream
    }

    start = () => {
        if (!this.micStream) {
            throw `Exception:${VOICE_CHANGER_CLIENT_EXCEPTION.ERR_MIC_STREAM_NOT_INITIALIZED}`
            return
        }
        this.micStream.playRecording()
        this._isVoiceChanging = true
    }
    stop = () => {
        if (!this.micStream) { return }
        this.micStream.pauseRecording()
        this._isVoiceChanging = false
    }
    get isVoiceChanging(): boolean {
        return this._isVoiceChanging
    }
    // Audio Streamer Settingg
    setServerUrl = (serverUrl: string, openTab: boolean = false) => {
        const url = validateUrl(serverUrl)
        const pageUrl = `${location.protocol}//${location.host}`

        if (url != pageUrl && url.length != 0 && location.protocol == "https:" && this.sslCertified.includes(url) == false) {
            if (openTab) {
                const value = window.confirm("MMVC Server is different from this page's origin. Open tab to open ssl connection. OK? (You can close the opened tab after ssl connection succeed.)");
                if (value) {
                    window.open(url, '_blank')
                    this.sslCertified.push(url)
                } else {
                    alert("Your voice conversion may fail...")
                }
            }
        }
        this.audioStreamer.setServerUrl(url)
        this.configurator.setServerUrl(url)
    }

    setProtocol = (mode: Protocol) => {
        this.audioStreamer.setProtocol(mode)
    }

    setInputChunkNum = (num: number) => {
        this.audioStreamer.setInputChunkNum(num)
    }

    setVoiceChangerMode = (val: VoiceChangerMode) => {
        this.audioStreamer.setVoiceChangerMode(val)
    }

    // configure worklet
    configureWorklet = (setting: WorkletSetting) => {
        const req: VoiceChangerWorkletProcessorRequest = {
            requestType: "config",
            voice: new ArrayBuffer(1),
            numTrancateTreshold: setting.numTrancateTreshold,
            volTrancateThreshold: setting.volTrancateThreshold,
            volTrancateLength: setting.volTrancateLength
        }
        this.vcNode.postReceivedVoice(req)
    }

    // Configurator Method
    uploadFile = (buf: ArrayBuffer, filename: string, onprogress: (progress: number, end: boolean) => void) => {
        return this.configurator.uploadFile(buf, filename, onprogress)
    }
    concatUploadedFile = (filename: string, chunkNum: number) => {
        return this.configurator.concatUploadedFile(filename, chunkNum)
    }
    loadModel = (configFilename: string, pyTorchModelFilename: string | null, onnxModelFilename: string | null) => {
        return this.configurator.loadModel(configFilename, pyTorchModelFilename, onnxModelFilename)
    }
    updateServerSettings = (key: ServerSettingKey, val: string) => {
        return this.configurator.updateSettings(key, val)
    }

    // Information
    getClientSettings = () => {
        return this.audioStreamer.getSettings()
    }
    getServerSettings = () => {
        return this.configurator.getSettings()
    }



}