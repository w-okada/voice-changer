import { BufferSize, createDummyMediaStream, Protocol, VoiceChangerMode, VoiceChangerRequestParamas, VoiceChnagerClient } from "@dannadori/voice-changer-client-js"
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
    start: (mmvcServerUrl: string, protocol: Protocol) => Promise<void>;
    stop: () => Promise<void>;
    changeInput: (audioInput: MediaStream | string, bufferSize: BufferSize, vfForceDisable: boolean) => Promise<void>
    changeInputChunkNum: (inputChunkNum: number) => void
    changeVoiceChangeMode: (voiceChangerMode: VoiceChangerMode) => void
    changeRequestParams: (params: VoiceChangerRequestParamas) => void
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
            console.log("client initialized!!")
            setClientInitialized(true)

            const audio = document.getElementById(props.audioOutputElementId) as HTMLAudioElement
            audio.srcObject = voiceChangerClientRef.current.stream
            audio.play()
        }
        initialized()
    }, [props.audioContext])

    const start = useMemo(() => {
        return async (mmvcServerUrl: string, protocol: Protocol) => {
            if (!voiceChangerClientRef.current) {
                console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setServerUrl(mmvcServerUrl, protocol, true)
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

    const changeInput = useMemo(() => {
        return async (audioInput: MediaStream | string, bufferSize: BufferSize, vfForceDisable: boolean) => {
            if (!voiceChangerClientRef.current || !props.audioContext) {
                console.log("not initialized", voiceChangerClientRef.current, props.audioContext)
                return
            }
            if (!audioInput || audioInput == "none") {
                console.log("setup! 1")
                const ms = createDummyMediaStream(props.audioContext)
                await voiceChangerClientRef.current.setup(ms, bufferSize, vfForceDisable)

            } else {
                console.log("setup! 2")
                await voiceChangerClientRef.current.setup(audioInput, bufferSize, vfForceDisable)
            }
        }
    }, [props.audioContext])


    const changeInputChunkNum = useMemo(() => {
        return (inputChunkNum: number) => {
            if (!voiceChangerClientRef.current) {
                // console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setInputChunkNum(inputChunkNum)
        }
    }, [])

    const changeVoiceChangeMode = useMemo(() => {
        return (voiceChangerMode: VoiceChangerMode) => {
            if (!voiceChangerClientRef.current) {
                // console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setVoiceChangerMode(voiceChangerMode)
        }
    }, [])

    const changeRequestParams = useMemo(() => {
        return (params: VoiceChangerRequestParamas) => {
            if (!voiceChangerClientRef.current) {
                // console.log("client is not initialized")
                return
            }
            voiceChangerClientRef.current.setRequestParams(params)
        }
    }, [])




    return {
        clientInitialized,
        bufferingTime,
        responseTime,
        volume,

        start,
        stop,
        changeInput,
        changeInputChunkNum,
        changeVoiceChangeMode,
        changeRequestParams,
    }
}