import { VoiceChangerWorkletNode, VoiceChangerWorkletListener } from "./VoiceChangerWorkletNode";
// @ts-ignore
import workerjs from "raw-loader!../worklet/dist/index.js";
import { VoiceFocusDeviceTransformer, VoiceFocusTransformDevice } from "amazon-chime-sdk-js";
import { createDummyMediaStream, validateUrl } from "./util";
import { AudioStreamerSetting, DefaultVoiceChangerClientSetting, ServerSettingKey, VoiceChangerClientSetting, VOICE_CHANGER_CLIENT_EXCEPTION, WorkletSetting } from "./const";
import MicrophoneStream from "microphone-stream";
import { AudioStreamer, Callbacks, AudioStreamerListeners } from "./AudioStreamer";
import { ServerConfigurator } from "./ServerConfigurator";

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
    private inputGainNode: GainNode | null = null
    private outputGainNode: GainNode | null = null
    private micStream: MicrophoneStream | null = null
    private audioStreamer!: AudioStreamer
    private vcNode!: VoiceChangerWorkletNode
    private currentMediaStreamAudioDestinationNode!: MediaStreamAudioDestinationNode


    private promiseForInitialize: Promise<void>
    private _isVoiceChanging = false

    private setting: VoiceChangerClientSetting = DefaultVoiceChangerClientSetting

    private sslCertified: string[] = []

    private sem = new BlockingQueue<number>();

    private callbacks: Callbacks = {
        onVoiceReceived: (data: ArrayBuffer): void => {
            this.vcNode.postReceivedVoice(data)
        }
    }

    constructor(ctx: AudioContext, vfEnable: boolean, audioStreamerListeners: AudioStreamerListeners, voiceChangerWorkletListener: VoiceChangerWorkletListener) {
        this.sem.enqueue(0);
        this.configurator = new ServerConfigurator()
        this.ctx = ctx
        this.vfEnable = vfEnable
        this.promiseForInitialize = new Promise<void>(async (resolve) => {
            const scriptUrl = URL.createObjectURL(new Blob([workerjs], { type: "text/javascript" }));
            await this.ctx.audioWorklet.addModule(scriptUrl)

            this.vcNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node 
            this.currentMediaStreamAudioDestinationNode = this.ctx.createMediaStreamDestination() // output node
            this.outputGainNode = this.ctx.createGain()
            this.outputGainNode.gain.value = this.setting.outputGain
            this.vcNode.connect(this.outputGainNode) // vc node -> output node
            this.outputGainNode.connect(this.currentMediaStreamAudioDestinationNode)
            // (vc nodeにはaudio streamerのcallbackでデータが投げ込まれる)
            this.audioStreamer = new AudioStreamer(this.callbacks, audioStreamerListeners, { objectMode: true, })

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

    /////////////////////////////////////////////////////
    // オペレーション
    /////////////////////////////////////////////////////
    /// Operations ///
    // setup = async (input: string | MediaStream | null, bufferSize: BufferSize, echoCancel: boolean = true, noiseSuppression: boolean = true, noiseSuppression2: boolean = false) => {
    setup = async () => {
        const lockNum = await this.lock()

        console.log(`Input Setup=> echo: ${this.setting.echoCancel}, noise1: ${this.setting.noiseSuppression}, noise2: ${this.setting.noiseSuppression2}`)
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

        //// Input デバイスがnullの時はmicStreamを止めてリターン
        if (!this.setting.audioInput) {
            console.log(`Input Setup=> client mic is disabled.`)
            if (this.micStream) {
                this.micStream.pauseRecording()
            }
            await this.unlock(lockNum)
            return
        }

        if (typeof this.setting.audioInput == "string") {
            this.currentMediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: this.setting.audioInput,
                    channelCount: 1,
                    sampleRate: this.setting.sampleRate,
                    sampleSize: 16,
                    autoGainControl: false,
                    echoCancellation: this.setting.echoCancel,
                    noiseSuppression: this.setting.noiseSuppression
                }
            })
            // this.currentMediaStream.getAudioTracks().forEach((x) => {
            //     console.log("MIC Setting(cap)", x.getCapabilities())
            //     console.log("MIC Setting(const)", x.getConstraints())
            //     console.log("MIC Setting(setting)", x.getSettings())
            // })
        } else {
            this.currentMediaStream = this.setting.audioInput
        }

        // create mic stream
        if (this.micStream) {
            this.micStream.unpipe()
            this.micStream.destroy()
            this.micStream = null
        }
        this.micStream = new MicrophoneStream({
            objectMode: true,
            bufferSize: this.setting.bufferSize,
            context: this.ctx
        })
        // connect nodes.
        this.currentMediaStreamAudioSourceNode = this.ctx.createMediaStreamSource(this.currentMediaStream)
        this.inputGainNode = this.ctx.createGain()
        this.inputGainNode.gain.value = this.setting.inputGain
        this.currentMediaStreamAudioSourceNode.connect(this.inputGainNode)
        if (this.currentDevice && this.setting.noiseSuppression2) {
            this.currentDevice.chooseNewInnerDevice(this.currentMediaStream)
            const voiceFocusNode = await this.currentDevice.createAudioNode(this.ctx); // vf node
            this.inputGainNode.connect(voiceFocusNode.start) // input node -> vf node
            voiceFocusNode.end.connect(this.outputNodeFromVF!)
            // this.micStream.setStream(this.outputNodeFromVF!.stream) // vf node -> mic stream
        } else {
            // const inputDestinationNodeForMicStream = this.ctx.createMediaStreamDestination()
            // this.inputGainNode.connect(inputDestinationNodeForMicStream)
            console.log("input___ media stream", this.currentMediaStream)
            this.currentMediaStream.getTracks().forEach(x => {
                console.log("input___ media stream set", x.getSettings())
                console.log("input___ media stream con", x.getConstraints())
                console.log("input___ media stream cap", x.getCapabilities())
            })
            console.log("input___ media node", this.currentMediaStreamAudioSourceNode)
            console.log("input___ gain node", this.inputGainNode.channelCount, this.inputGainNode)
            this.inputGainNode.connect(this.vcNode)




            // this.micStream.setStream(inputDestinationNodeForMicStream.stream) // input device -> mic stream
        }
        this.micStream.pipe(this.audioStreamer) // mic stream -> audio streamer
        if (!this._isVoiceChanging) {
            this.micStream.pauseRecording()
        } else {
            this.micStream.playRecording()
        }
        console.log("Input Setup=> success")
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

    ////////////////////////
    /// 設定
    //////////////////////////////
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
        this.audioStreamer.updateSetting({ ...this.audioStreamer.getSettings(), serverUrl: url })
        this.configurator.setServerUrl(url)
    }

    updateClientSetting = (setting: VoiceChangerClientSetting) => {
        console.log(`[VoiceChangerClient] Updating Client Setting,`, this.setting, setting)
        let reconstructInputRequired = false
        if (
            this.setting.audioInput != setting.audioInput ||
            this.setting.bufferSize != setting.bufferSize ||
            this.setting.echoCancel != setting.echoCancel ||
            this.setting.noiseSuppression != setting.noiseSuppression ||
            this.setting.noiseSuppression2 != setting.noiseSuppression2 ||
            this.setting.sampleRate != setting.sampleRate
        ) {
            reconstructInputRequired = true
        }

        if (this.setting.inputGain != setting.inputGain) {
            this.setInputGain(setting.inputGain)
        }
        if (this.setting.outputGain != setting.outputGain) {
            this.setOutputGain(setting.outputGain)
        }

        this.setting = setting
        if (reconstructInputRequired) {
            this.setup()
        }



    }

    setInputGain = (val: number) => {
        this.setting.inputGain = val
        if (!this.inputGainNode) {
            return
        }
        this.inputGainNode.gain.value = val
    }

    setOutputGain = (val: number) => {
        if (!this.outputGainNode) {
            return
        }
        this.outputGainNode.gain.value = val
    }

    /////////////////////////////////////////////////////
    // コンポーネント設定、操作
    /////////////////////////////////////////////////////
    //##  Server ##//
    updateServerSettings = (key: ServerSettingKey, val: string) => {
        return this.configurator.updateSettings(key, val)
    }
    uploadFile = (buf: ArrayBuffer, filename: string, onprogress: (progress: number, end: boolean) => void) => {
        return this.configurator.uploadFile(buf, filename, onprogress)
    }
    concatUploadedFile = (filename: string, chunkNum: number) => {
        return this.configurator.concatUploadedFile(filename, chunkNum)
    }
    loadModel = (configFilename: string, pyTorchModelFilename: string | null, onnxModelFilename: string | null) => {
        return this.configurator.loadModel(configFilename, pyTorchModelFilename, onnxModelFilename)
    }

    //##  Worklet ##//
    configureWorklet = (setting: WorkletSetting) => {
        this.vcNode.configure(setting)
    }
    startOutputRecordingWorklet = () => {
        this.vcNode.startOutputRecordingWorklet()
    }
    stopOutputRecordingWorklet = () => {
        this.vcNode.stopOutputRecordingWorklet()
    }


    //##  Audio Streamer ##//
    updateAudioStreamerSetting = (setting: AudioStreamerSetting) => {
        this.audioStreamer.updateSetting(setting)
    }


    /////////////////////////////////////////////////////
    // 情報取得
    /////////////////////////////////////////////////////
    // Information
    getClientSettings = () => {
        return this.audioStreamer.getSettings()
    }
    getServerSettings = () => {
        return this.configurator.getSettings()
    }


    getSocketId = () => {
        return this.audioStreamer.getSocketId()
    }


}