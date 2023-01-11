import { ServerInfo, BufferSize, createDummyMediaStream, DefaultVoiceChangerOptions, DefaultVoiceChangerRequestParamas, Framework, OnnxExecutionProvider, Protocol, SampleRate, ServerSettingKey, Speaker, VoiceChangerMode, VoiceChnagerClient } from "@dannadori/voice-changer-client-js"
import { useEffect, useMemo, useRef, useState } from "react"

export type UseClientProps = {
    audioContext: AudioContext | null
    audioOutputElementId: string
}

export type SettingState = {
    // server setting
    mmvcServerUrl: string
    pyTorchModel: File | null
    configFile: File | null
    onnxModel: File | null
    protocol: Protocol
    framework: Framework
    onnxExecutionProvider: OnnxExecutionProvider

    // device setting
    audioInput: string | MediaStream | null;
    sampleRate: SampleRate;

    // speaker setting
    speakers: Speaker[]
    editSpeakerTargetId: number
    editSpeakerTargetName: string
    srcId: number
    dstId: number

    // convert setting
    bufferSize: BufferSize
    inputChunkNum: number
    convertChunkNum: number
    gpu: number
    crossFadeOffsetRate: number
    crossFadeEndRate: number

    // advanced setting
    vfForceDisabled: boolean
    voiceChangerMode: VoiceChangerMode
}

const InitialSettingState: SettingState = {
    mmvcServerUrl: DefaultVoiceChangerOptions.mmvcServerUrl,
    pyTorchModel: null,
    configFile: null,
    onnxModel: null,
    protocol: DefaultVoiceChangerOptions.protocol,
    framework: DefaultVoiceChangerOptions.framework,
    onnxExecutionProvider: DefaultVoiceChangerOptions.onnxExecutionProvider,

    audioInput: "none",
    sampleRate: DefaultVoiceChangerOptions.sampleRate,

    speakers: DefaultVoiceChangerOptions.speakers,
    editSpeakerTargetId: 0,
    editSpeakerTargetName: "",
    srcId: DefaultVoiceChangerRequestParamas.srcId,
    dstId: DefaultVoiceChangerRequestParamas.dstId,

    bufferSize: DefaultVoiceChangerOptions.bufferSize,
    inputChunkNum: DefaultVoiceChangerOptions.inputChunkNum,
    convertChunkNum: DefaultVoiceChangerRequestParamas.convertChunkNum,
    gpu: DefaultVoiceChangerRequestParamas.gpu,
    crossFadeOffsetRate: DefaultVoiceChangerRequestParamas.crossFadeOffsetRate,
    crossFadeEndRate: DefaultVoiceChangerRequestParamas.crossFadeEndRate,
    vfForceDisabled: DefaultVoiceChangerOptions.forceVfDisable,
    voiceChangerMode: DefaultVoiceChangerOptions.voiceChangerMode
}

export type ClientState = {
    clientInitialized: boolean
    bufferingTime: number;
    responseTime: number;
    volume: number;
    uploadProgress: number;
    isUploading: boolean

    // Setting
    settingState: SettingState
    serverInfo: ServerInfo | undefined
    setSettingState: (setting: SettingState) => void

    // Client Control
    loadModel: () => Promise<void>
    start: () => Promise<void>;
    stop: () => Promise<void>;
    getInfo: () => Promise<void>
}



export const useClient = (props: UseClientProps): ClientState => {

    // (1) クライアント初期化
    const voiceChangerClientRef = useRef<VoiceChnagerClient | null>(null)
    const [clientInitialized, setClientInitialized] = useState<boolean>(false)
    const initializedResolveRef = useRef<(value: void | PromiseLike<void>) => void>()
    const initializedPromise = useMemo(() => {
        return new Promise<void>((resolve) => {
            initializedResolveRef.current = resolve
        })
    }, [])
    const [bufferingTime, setBufferingTime] = useState<number>(0)
    const [responseTime, setResponseTime] = useState<number>(0)
    const [volume, setVolume] = useState<number>(0)


    // Colab対応
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const colab = params.get("colab")
        if (colab == "true") {

        }
    }, [])

    useEffect(() => {
        const initialized = async () => {
            if (!props.audioContext) {
                return
            }
            const voiceChangerClient = new VoiceChnagerClient(props.audioContext, true, {
                notifySendBufferingTime: (val: number) => {
                    setBufferingTime(val)
                },
                notifyResponseTime: (val: number) => {
                    setResponseTime(val)
                },
                notifyException: (mes: string) => {
                    if (mes.length > 0) {
                        console.log(`error:${mes}`)
                    }
                }
            }, {
                notifyVolume: (vol: number) => {
                    setVolume(vol)
                }
            })

            await voiceChangerClient.isInitialized()
            voiceChangerClientRef.current = voiceChangerClient
            console.log("[useClient] client initialized")
            setClientInitialized(true)

            const audio = document.getElementById(props.audioOutputElementId) as HTMLAudioElement
            audio.srcObject = voiceChangerClientRef.current.stream
            audio.play()
            initializedResolveRef.current!()
        }
        initialized()
    }, [props.audioContext])



    // (2) 設定
    const [settingState, setSettingState] = useState<SettingState>(InitialSettingState)
    const [displaySettingState, setDisplaySettingState] = useState<SettingState>(InitialSettingState)
    const [serverInfo, setServerInfo] = useState<ServerInfo>()
    const [uploadProgress, setUploadProgress] = useState<number>(0)
    const [isUploading, setIsUploading] = useState<boolean>(false)

    // (2-1) server setting
    // (a) サーバURL設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            voiceChangerClientRef.current!.setServerUrl(settingState.mmvcServerUrl, true)
            voiceChangerClientRef.current!.stop()
            getInfo()

        })()
    }, [settingState.mmvcServerUrl])
    // (b) プロトコル設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            voiceChangerClientRef.current!.setProtocol(settingState.protocol)
        })()
    }, [settingState.protocol])
    // (c) フレームワーク設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.framework, "" + settingState.framework)
            setServerInfo(info)

        })()
    }, [settingState.framework])
    // (d) OnnxExecutionProvider設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.onnxExecutionProvider, settingState.onnxExecutionProvider)
            setServerInfo(info)

        })()
    }, [settingState.onnxExecutionProvider])

    // (e) モデルアップロード
    const uploadFile = useMemo(() => {
        return async (file: File, onprogress: (progress: number, end: boolean) => void) => {
            await initializedPromise
            const num = await voiceChangerClientRef.current!.uploadFile(file, onprogress)
            const res = await voiceChangerClientRef.current!.concatUploadedFile(file, num)
            console.log("uploaded", num, res)
        }
    }, [])
    const loadModel = useMemo(() => {
        return async () => {
            if (!settingState.pyTorchModel && !settingState.onnxModel) {
                alert("PyTorchモデルとONNXモデルのどちらか一つ以上指定する必要があります。")
                return
            }
            if (!settingState.configFile) {
                alert("Configファイルを指定する必要があります。")
                return
            }
            await initializedPromise
            setUploadProgress(0)
            setIsUploading(true)
            const models = [settingState.pyTorchModel, settingState.onnxModel].filter(x => { return x != null }) as File[]
            for (let i = 0; i < models.length; i++) {
                const progRate = 1 / models.length
                const progOffset = 100 * i * progRate
                await uploadFile(models[i], (progress: number, end: boolean) => {
                    // console.log(progress * progRate + progOffset, end, progRate,)
                    setUploadProgress(progress * progRate + progOffset)
                })
            }

            await uploadFile(settingState.configFile, (progress: number, end: boolean) => {
                console.log(progress, end)
            })

            const serverInfo = await voiceChangerClientRef.current!.loadModel(settingState.configFile, settingState.pyTorchModel, settingState.onnxModel)
            console.log(serverInfo)
            setUploadProgress(0)
            setIsUploading(false)
        }
    }, [settingState.pyTorchModel, settingState.onnxModel, settingState.configFile])

    // (2-2) device setting
    // (a) インプット設定。audio nodes の設定の都合上、バッファサイズの変更も併せて反映させる。
    useEffect(() => {
        (async () => {
            await initializedPromise
            if (!settingState.audioInput || settingState.audioInput == "none") {
                console.log("[useClient] setup!(1)", settingState.audioInput)
                const ms = createDummyMediaStream(props.audioContext!)
                await voiceChangerClientRef.current!.setup(ms, settingState.bufferSize, settingState.vfForceDisabled)

            } else {
                console.log("[useClient] setup!(2)", settingState.audioInput)
                await voiceChangerClientRef.current!.setup(settingState.audioInput, settingState.bufferSize, settingState.vfForceDisabled)
            }
        })()
    }, [settingState.audioInput, settingState.bufferSize, settingState.vfForceDisabled])


    // (2-3) speaker setting
    // (a) srcId設定。
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.srcId, "" + settingState.srcId)
            setServerInfo(info)

        })()
    }, [settingState.srcId])

    // (b) dstId設定。
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.dstId, "" + settingState.dstId)
            setServerInfo(info)

        })()
    }, [settingState.dstId])


    // (2-4) convert setting
    // (a) input chunk num設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            voiceChangerClientRef.current!.setInputChunkNum(settingState.inputChunkNum)
        })()
    }, [settingState.inputChunkNum])

    // (b) convert chunk num設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.convertChunkNum, "" + settingState.convertChunkNum)
            setServerInfo(info)
        })()
    }, [settingState.convertChunkNum])

    // (c) gpu設定
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.gpu, "" + settingState.gpu)
            setServerInfo(info)
        })()
    }, [settingState.gpu])

    // (d) crossfade設定1
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.crossFadeOffsetRate, "" + settingState.crossFadeOffsetRate)
            setServerInfo(info)
        })()
    }, [settingState.crossFadeOffsetRate])

    // (e) crossfade設定2
    useEffect(() => {
        (async () => {
            await initializedPromise
            const info = await voiceChangerClientRef.current!.updateServerSettings(ServerSettingKey.crossFadeEndRate, "" + settingState.crossFadeEndRate)
            setServerInfo(info)
        })()
    }, [settingState.crossFadeEndRate])

    // (2-5) advanced setting
    //// VFDisableはinput設定で合わせて設定。
    // (a) voice changer mode
    useEffect(() => {
        (async () => {
            await initializedPromise
            voiceChangerClientRef.current!.setVoiceChangerMode(settingState.voiceChangerMode)
            voiceChangerClientRef.current!.stop()
        })()
    }, [settingState.voiceChangerMode])

    // (2-6) server control
    // (1) start
    const start = useMemo(() => {
        return async () => {
            await initializedPromise
            voiceChangerClientRef.current!.setServerUrl(settingState.mmvcServerUrl, true)
            voiceChangerClientRef.current!.start()
        }
    }, [settingState.mmvcServerUrl])
    // (2) stop
    const stop = useMemo(() => {
        return async () => {
            await initializedPromise
            voiceChangerClientRef.current!.stop()
        }
    }, [])

    // (3) get info
    const getInfo = useMemo(() => {
        return async () => {
            await initializedPromise
            const serverSettings = await voiceChangerClientRef.current!.getServerSettings()
            const clientSettings = await voiceChangerClientRef.current!.getClientSettings()
            setServerInfo(serverSettings)
            console.log(serverSettings, clientSettings)
        }
    }, [])

    // (x)
    useEffect(() => {
        if (serverInfo && serverInfo.status == "OK") {
            setDisplaySettingState({
                ...settingState,
                convertChunkNum: serverInfo.convertChunkNum,
                crossFadeOffsetRate: serverInfo.crossFadeOffsetRate,
                crossFadeEndRate: serverInfo.crossFadeEndRate,
                gpu: serverInfo.gpu,
                srcId: serverInfo.srcId,
                dstId: serverInfo.dstId,
                framework: serverInfo.framework,
                onnxExecutionProvider: serverInfo.providers.length > 0 ? serverInfo.providers[0] as OnnxExecutionProvider : "CPUExecutionProvider"
            })
        } else {
            setDisplaySettingState({
                ...settingState,
            })
        }

    }, [settingState, serverInfo])


    // Colab対応
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const colab = params.get("colab")
        if (colab == "true") {
            setSettingState({
                ...settingState,
                protocol: "rest"
            })
        }
    }, [])


    return {
        clientInitialized,
        bufferingTime,
        responseTime,
        volume,
        uploadProgress,
        isUploading,

        settingState: displaySettingState,
        serverInfo,
        setSettingState,
        loadModel,
        start,
        stop,
        getInfo,
    }
}