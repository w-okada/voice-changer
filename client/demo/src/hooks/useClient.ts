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
        }
        initialized()
    }, [props.audioContext])

    // Client Setting
    const setServerUrl = useMemo(() => {
        return async (mmvcServerUrl: string) => {
            if (!voiceChangerClientRef.current) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setServerUrl(mmvcServerUrl, true)
            voiceChangerClientRef.current.stop()
        }
    }, [])

    const setProtocol = useMemo(() => {
        return async (protocol: Protocol) => {
            if (!voiceChangerClientRef.current) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setProtocol(protocol)
        }
    }, [])

    const setInputChunkNum = useMemo(() => {
        return async (num: number) => {
            if (!voiceChangerClientRef.current) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setInputChunkNum(num)
        }
    }, [])

    const setVoiceChangerMode = useMemo(() => {
        return async (val: VoiceChangerMode) => {
            if (!voiceChangerClientRef.current) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setVoiceChangerMode(val)
            voiceChangerClientRef.current.stop()
        }
    }, [])


    // Client Control
    const start = useMemo(() => {
        return async (mmvcServerUrl: string) => {
            if (!voiceChangerClientRef.current) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setServerUrl(mmvcServerUrl, true)
            voiceChangerClientRef.current.start()
        }
    }, [])
    const stop = useMemo(() => {
        return async () => {
            if (!voiceChangerClientRef.current) {
                // console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.stop()
        }
    }, [])


    // Device Setting
    const changeInput = useMemo(() => {
        return async (audioInput: MediaStream | string, bufferSize: BufferSize, vfForceDisable: boolean) => {
            if (!voiceChangerClientRef.current || !props.audioContext) {
                console.log("[useClient] not initialized", voiceChangerClientRef.current, props.audioContext)
                return
            }
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
            if (!voiceChangerClientRef.current) {
                throw "[useClient] Client Not Initialized."
            }
            const num = await voiceChangerClientRef.current.uploadFile(file, onprogress)
            const res = await voiceChangerClientRef.current.concatUploadedFile(file, num)
            console.log("upload", num, res)
        }
    }, [])

    const loadModel = useMemo(() => {
        return async (configFile: File, pyTorchModelFile: File | null, onnxModelFile: File | null) => {
            if (!voiceChangerClientRef.current) {
                throw "[useClient] Client Not Initialized."
            }
            await voiceChangerClientRef.current.loadModel(configFile, pyTorchModelFile, onnxModelFile)
            console.log("load model")
        }
    }, [])

    const updateSettings = useMemo(() => {
        return async (key: ServerSettingKey, val: string | number) => {
            if (!voiceChangerClientRef.current) {
                throw "[useClient] Client Not Initialized."
            }
            return await voiceChangerClientRef.current.updateServerSettings(key, "" + val)
        }
    }, [])

    // Information
    const getInfo = useMemo(() => {
        return async () => {
            if (!voiceChangerClientRef.current) {
                throw "[useClient] Client Not Initialized."
            }
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