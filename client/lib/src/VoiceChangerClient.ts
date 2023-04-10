import { VoiceChangerWorkletNode, VoiceChangerWorkletListener } from "./VoiceChangerWorkletNode";
// @ts-ignore
import workerjs from "raw-loader!../worklet/dist/index.js";
import { VoiceFocusDeviceTransformer, VoiceFocusTransformDevice } from "amazon-chime-sdk-js";
import { createDummyMediaStream, validateUrl } from "./util";
import { ClientType, DefaultVoiceChangerClientSetting, ServerSettingKey, VoiceChangerClientSetting, WorkletNodeSetting, WorkletSetting } from "./const";
import { ServerConfigurator } from "./ServerConfigurator";

// オーディオデータの流れ
// input node(mic or MediaStream) -> [vf node] -> [vc node] -> 
//    sio/rest server ->  [vc node] -> output node

import { BlockingQueue } from "./utils/BlockingQueue";

export class VoiceChangerClient {
    private configurator: ServerConfigurator
    private ctx: AudioContext
    private vfEnable = false
    private vf: VoiceFocusDeviceTransformer | null = null
    private currentDevice: VoiceFocusTransformDevice | null = null

    private currentMediaStream: MediaStream | null = null
    private currentMediaStreamAudioSourceNode: MediaStreamAudioSourceNode | null = null
    private inputGainNode: GainNode | null = null
    private outputGainNode: GainNode | null = null
    private vcInNode!: VoiceChangerWorkletNode
    private vcOutNode!: VoiceChangerWorkletNode
    private currentMediaStreamAudioDestinationNode!: MediaStreamAudioDestinationNode


    private promiseForInitialize: Promise<void>
    private _isVoiceChanging = false

    private setting: VoiceChangerClientSetting = DefaultVoiceChangerClientSetting

    private sslCertified: string[] = []

    private sem = new BlockingQueue<number>();

    constructor(ctx: AudioContext, vfEnable: boolean, voiceChangerWorkletListener: VoiceChangerWorkletListener) {
        this.sem.enqueue(0);
        this.configurator = new ServerConfigurator()
        this.ctx = ctx
        this.vfEnable = vfEnable
        this.promiseForInitialize = new Promise<void>(async (resolve) => {
            const scriptUrl = URL.createObjectURL(new Blob([workerjs], { type: "text/javascript" }));

            // await this.ctx.audioWorklet.addModule(scriptUrl)
            // this.vcInNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node 

            try {
                this.vcInNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node 
            } catch (err) {
                await this.ctx.audioWorklet.addModule(scriptUrl)
                this.vcInNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node 
            }



            // const ctx44k = new AudioContext({ sampleRate: 44100 }) // これでもプチプチが残る
            const ctx44k = new AudioContext({ sampleRate: 48000 }) // 結局これが一番まし。
            console.log("audio out:", ctx44k)
            try {
                this.vcOutNode = new VoiceChangerWorkletNode(ctx44k, voiceChangerWorkletListener); // vc node 
            } catch (err) {
                await ctx44k.audioWorklet.addModule(scriptUrl)
                this.vcOutNode = new VoiceChangerWorkletNode(ctx44k, voiceChangerWorkletListener); // vc node 
            }
            this.currentMediaStreamAudioDestinationNode = ctx44k.createMediaStreamDestination() // output node
            this.outputGainNode = ctx44k.createGain()
            this.outputGainNode.gain.value = this.setting.outputGain
            this.vcOutNode.connect(this.outputGainNode) // vc node -> output node
            this.outputGainNode.connect(this.currentMediaStreamAudioDestinationNode)

            if (this.vfEnable) {
                this.vf = await VoiceFocusDeviceTransformer.create({ variant: 'c20' })
                const dummyMediaStream = createDummyMediaStream(this.ctx)
                this.currentDevice = (await this.vf.createTransformDevice(dummyMediaStream)) || null;
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
    setup = async () => {
        const lockNum = await this.lock()

        console.log(`Input Setup=> echo: ${this.setting.echoCancel}, noise1: ${this.setting.noiseSuppression}, noise2: ${this.setting.noiseSuppression2}`)
        // condition check
        if (!this.vcInNode) {
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
            this.vcInNode.stop()
            await this.unlock(lockNum)
            return
        }

        if (typeof this.setting.audioInput == "string") {
            try {
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

            } catch (e) {
                console.warn(e)
                throw e
            }
            // this.currentMediaStream.getAudioTracks().forEach((x) => {
            //     console.log("MIC Setting(cap)", x.getCapabilities())
            //     console.log("MIC Setting(const)", x.getConstraints())
            //     console.log("MIC Setting(setting)", x.getSettings())
            // })
        } else {
            this.currentMediaStream = this.setting.audioInput
        }

        // connect nodes.
        this.currentMediaStreamAudioSourceNode = this.ctx.createMediaStreamSource(this.currentMediaStream)
        this.inputGainNode = this.ctx.createGain()
        this.inputGainNode.gain.value = this.setting.inputGain
        this.currentMediaStreamAudioSourceNode.connect(this.inputGainNode)
        if (this.currentDevice && this.setting.noiseSuppression2) {
            this.currentDevice.chooseNewInnerDevice(this.currentMediaStream)
            const voiceFocusNode = await this.currentDevice.createAudioNode(this.ctx); // vf node
            this.inputGainNode.connect(voiceFocusNode.start) // input node -> vf node
            voiceFocusNode.end.connect(this.vcInNode)
        } else {
            // console.log("input___ media stream", this.currentMediaStream)
            // this.currentMediaStream.getTracks().forEach(x => {
            //     console.log("input___ media stream set", x.getSettings())
            //     console.log("input___ media stream con", x.getConstraints())
            //     console.log("input___ media stream cap", x.getCapabilities())
            // })
            // console.log("input___ media node", this.currentMediaStreamAudioSourceNode)
            // console.log("input___ gain node", this.inputGainNode.channelCount, this.inputGainNode)
            this.inputGainNode.connect(this.vcInNode)
        }
        this.vcInNode.setOutputNode(this.vcOutNode)
        console.log("Input Setup=> success")
        await this.unlock(lockNum)
    }
    get stream(): MediaStream {
        return this.currentMediaStreamAudioDestinationNode.stream
    }

    start = () => {
        this.vcInNode.start()
        this._isVoiceChanging = true
    }
    stop = () => {
        this.vcInNode.stop()
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
        this.vcInNode.updateSetting({ ...this.vcInNode.getSettings(), serverUrl: url })
        this.configurator.setServerUrl(url)
    }

    updateClientSetting = (setting: VoiceChangerClientSetting) => {
        console.log(`[VoiceChangerClient] Updating Client Setting,`, this.setting, setting)
        let reconstructInputRequired = false
        if (
            this.setting.audioInput != setting.audioInput ||
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
    switchModelType = (clientType: ClientType) => {
        return this.configurator.switchModelType(clientType)
    }
    getModelType = () => {
        return this.configurator.getModelType()
    }


    updateServerSettings = (key: ServerSettingKey, val: string) => {
        return this.configurator.updateSettings(key, val)
    }
    uploadFile = (buf: ArrayBuffer, filename: string, onprogress: (progress: number, end: boolean) => void) => {
        return this.configurator.uploadFile(buf, filename, onprogress)
    }
    concatUploadedFile = (filename: string, chunkNum: number) => {
        return this.configurator.concatUploadedFile(filename, chunkNum)
    }
    loadModel = (
        configFilename: string,
        pyTorchModelFilename: string | null,
        onnxModelFilename: string | null,
        clusterTorchModelFilename: string | null,
        featureFilename: string | null,
        indexFilename: string | null,
        isHalf: boolean
    ) => {
        // !! 注意!! hubertTorchModelは固定値で上書きされるため、設定しても効果ない。
        return this.configurator.loadModel(configFilename, pyTorchModelFilename, onnxModelFilename, clusterTorchModelFilename, featureFilename, indexFilename, isHalf)
    }

    //##  Worklet ##//
    configureWorklet = (setting: WorkletSetting) => {
        this.vcInNode.configure(setting)
        this.vcOutNode.configure(setting)
    }
    startOutputRecording = () => {
        this.vcOutNode.startOutputRecording()
    }
    stopOutputRecording = () => {
        return this.vcOutNode.stopOutputRecording()
    }
    trancateBuffer = () => {
        this.vcOutNode.trancateBuffer()
    }
    //##  Worklet Node ##//
    updateWorkletNodeSetting = (setting: WorkletNodeSetting) => {
        this.vcInNode.updateSetting(setting)
        this.vcOutNode.updateSetting(setting)
    }


    /////////////////////////////////////////////////////
    // 情報取得
    /////////////////////////////////////////////////////
    // Information
    getClientSettings = () => {
        return this.vcInNode.getSettings()
    }
    getServerSettings = () => {
        return this.configurator.getSettings()
    }

    getSocketId = () => {
        return this.vcInNode.getSocketId()
    }

}