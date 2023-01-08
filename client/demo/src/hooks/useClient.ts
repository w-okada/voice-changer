import { BufferSize, createDummyMediaStream, Protocol, ServerSettingKey, VoiceChangerMode, VoiceChnagerClient } from "@dannadori/voice-changer-client-js"
import { useEffect, useMemo, useRef, useState } from "react"

export type UseClientProps = {
    audioContext: AudioContext | null
    audioOutputElementId: string
}

export type ClientState = {
    clientInitialized: boolean
    bufferingTime: number;
    responseTime: number;
    volume: number;
    // Client Setting
    setServerUrl: (mmvcServerUrl: string) => Promise<void>
    setProtocol: (protocol: Protocol) => Promise<void>
    setInputChunkNum: (num: number) => Promise<void>
    setVoiceChangerMode: (val: VoiceChangerMode) => Promise<void>

    // Client Control
    start: (mmvcServerUrl: string, protocol: Protocol) => Promise<void>;
    stop: () => Promise<void>;

    // Device Setting
    changeInput: (audioInput: MediaStream | string, bufferSize: BufferSize, vfForceDisable: boolean) => Promise<void>

    // Server Setting
    uploadFile: (file: File, onprogress: (progress: number, end: boolean) => void) => Promise<void>
    loadModel: (configFile: File, pyTorchModelFile: File | null, onnxModelFile: File | null) => Promise<void>
    updateSettings: (key: ServerSettingKey, val: string | number) => Promise<any>

    // Information
    getInfo: () => Promise<void>
}
export const useClient = (props: UseClientProps): ClientState => {

    const voiceChangerClientRef = useRef<VoiceChnagerClient | null>(null)
    const [clientInitialized, setClientInitialized] = useState<boolean>(false)

    const [bufferingTime, setBufferingTime] = useState<number>(0)
    const [responseTime, setResponseTime] = useState<number>(0)
    const [volume, setVolume] = useState<number>(0)

    const initializedResolveRef = useRef<(value: void | PromiseLike<void>) => void>()
    const initializedPromise = useMemo(() => {
        return new Promise<void>((resolve) => {
            initializedResolveRef.current = resolve
        })
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

    // Client Setting
    const setServerUrl = useMemo(() => {
        return async (mmvcServerUrl: string) => {
            await initializedPromise
            voiceChangerClientRef.current.setServerUrl(mmvcServerUrl, true)
            voiceChangerClientRef.current.stop()
        }
    }, [])

    const setProtocol = useMemo(() => {
        return async (protocol: Protocol) => {
            await initializedPromise
            voiceChangerClientRef.current.setProtocol(protocol)
        }
    }, [])

    const setInputChunkNum = useMemo(() => {
        return async (num: number) => {
            await initializedPromise
            voiceChangerClientRef.current.setInputChunkNum(num)
        }
    }, [])

    const setVoiceChangerMode = useMemo(() => {
        return async (val: VoiceChangerMode) => {
            await initializedPromise
            voiceChangerClientRef.current.setVoiceChangerMode(val)
            voiceChangerClientRef.current.stop()
        }
    }, [])


    // Client Control
    const start = useMemo(() => {
        return async (mmvcServerUrl: string) => {
            await initializedPromise
            voiceChangerClientRef.current.setServerUrl(mmvcServerUrl, true)
            voiceChangerClientRef.current.start()
        }
    }, [])
    const stop = useMemo(() => {
        return async () => {
            await initializedPromise
            voiceChangerClientRef.current.stop()
        }
    }, [])


    // Device Setting
    const changeInput = useMemo(() => {
        return async (audioInput: MediaStream | string, bufferSize: BufferSize, vfForceDisable: boolean) => {
            await initializedPromise
            if (!props.audioContext) return
            if (!audioInput || audioInput == "none") {
                console.log("[useClient] setup!(1)", audioInput)
                const ms = createDummyMediaStream(props.audioContext)
                await voiceChangerClientRef.current.setup(ms, bufferSize, vfForceDisable)

            } else {
                console.log("[useClient] setup!(2)", audioInput)
                await voiceChangerClientRef.current.setup(audioInput, bufferSize, vfForceDisable)
            }
        }
    }, [props.audioContext])



    // Server Setting
    const uploadFile = useMemo(() => {
        return async (file: File, onprogress: (progress: number, end: boolean) => void) => {
            await initializedPromise
            const num = await voiceChangerClientRef.current.uploadFile(file, onprogress)
            const res = await voiceChangerClientRef.current.concatUploadedFile(file, num)
            console.log("uploaded", num, res)
        }
    }, [])

    const loadModel = useMemo(() => {
        return async (configFile: File, pyTorchModelFile: File | null, onnxModelFile: File | null) => {
            await initializedPromise
            await voiceChangerClientRef.current.loadModel(configFile, pyTorchModelFile, onnxModelFile)
            console.log("loaded model")
        }
    }, [])

    const updateSettings = useMemo(() => {
        return async (key: ServerSettingKey, val: string | number) => {
            await initializedPromise
            return await voiceChangerClientRef.current.updateServerSettings(key, "" + val)
        }
    }, [])

    // Information
    const getInfo = useMemo(() => {
        return async () => {
            await initializedPromise
            const serverSettings = await voiceChangerClientRef.current.getServerSettings()
            const clientSettings = await voiceChangerClientRef.current.getClientSettings()
            console.log(serverSettings, clientSettings)
        }
    }, [])


    return {
        clientInitialized,
        bufferingTime,
        responseTime,
        volume,

        setServerUrl,
        setProtocol,
        setInputChunkNum,
        setVoiceChangerMode,

        start,
        stop,

        changeInput,

        uploadFile,
        loadModel,
        updateSettings,

        getInfo,
    }
}