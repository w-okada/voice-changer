import { useEffect, useMemo, useRef, useState } from "react"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { ClientSettingState, useClientSetting } from "./useClientSetting"
import { IndexedDBStateAndMethod, useIndexedDB } from "./useIndexedDB"
import { ServerSettingState, useServerSetting } from "./useServerSetting"
import { useWorkletNodeSetting, WorkletNodeSettingState } from "./useWorkletNodeSetting"
import { useWorkletSetting, WorkletSettingState } from "./useWorkletSetting"
import { DefaultVoiceChangerClientSetting, DefaultWorkletNodeSetting, DefaultWorkletSetting } from "../const"

export type UseClientProps = {
    audioContext: AudioContext | null
}

export type ClientState = {
    initialized: boolean
    // 各種設定I/Fへの参照
    workletSetting: WorkletSettingState
    clientSetting: ClientSettingState
    workletNodeSetting: WorkletNodeSettingState
    serverSetting: ServerSettingState
    indexedDBState: IndexedDBStateAndMethod

    // モニタリングデータ
    bufferingTime: number;
    volume: number;
    performance: PerformanceData
    updatePerformance: (() => Promise<void>) | null
    // setClientType: (val: ClientType) => void

    // 情報取得
    getInfo: () => Promise<void>
    // 設定クリア
    clearSetting: () => Promise<void>
    // AudioOutputElement  設定
    setAudioOutputElementId: (elemId: string) => void

    ioErrorCount: number
    resetIoErrorCount: () => void
}

export type PerformanceData = {
    responseTime: number
    preprocessTime: number
    mainprocessTime: number
    postprocessTime: number
}
const InitialPerformanceData: PerformanceData = {
    responseTime: 0,
    preprocessTime: 0,
    mainprocessTime: 0,
    postprocessTime: 0
}

export const useClient = (props: UseClientProps): ClientState => {

    const [initialized, setInitialized] = useState<boolean>(false)
    // const [clientType, setClientType] = useState<ClientType | null>(null)
    // (1-1) クライアント    
    const voiceChangerClientRef = useRef<VoiceChangerClient | null>(null)
    const [voiceChangerClient, setVoiceChangerClient] = useState<VoiceChangerClient | null>(voiceChangerClientRef.current)
    //// クライアント初期化待ち用フラグ
    const initializedResolveRef = useRef<(value: void | PromiseLike<void>) => void>()
    const initializedPromise = useMemo(() => {
        return new Promise<void>((resolve) => {
            initializedResolveRef.current = resolve
        })
    }, [])


    // (1-2) 各種設定I/F
    const clientSetting = useClientSetting({ voiceChangerClient, audioContext: props.audioContext, defaultVoiceChangerClientSetting: DefaultVoiceChangerClientSetting })
    const workletNodeSetting = useWorkletNodeSetting({ voiceChangerClient: voiceChangerClient, defaultWorkletNodeSetting: DefaultWorkletNodeSetting })
    const workletSetting = useWorkletSetting({ voiceChangerClient, defaultWorkletSetting: DefaultWorkletSetting })
    const serverSetting = useServerSetting({ voiceChangerClient })
    const indexedDBState = useIndexedDB({ clientType: null })


    // (1-3) モニタリングデータ
    const [bufferingTime, setBufferingTime] = useState<number>(0)
    const [performance, setPerformance] = useState<PerformanceData>(InitialPerformanceData)
    const [volume, setVolume] = useState<number>(0)
    const [ioErrorCount, setIoErrorCount] = useState<number>(0)

    //// Server Audio Deviceを使うとき、モニタリングデータはpolling
    const updatePerformance = useMemo(() => {
        if (!voiceChangerClientRef.current) {
            return null
        }

        return async () => {
            if (voiceChangerClientRef.current) {
                const performance = await voiceChangerClientRef.current!.getPerformance()
                const responseTime = performance[0]
                const preprocessTime = performance[1]
                const mainprocessTime = performance[2]
                const postprocessTime = performance[3]
                setPerformance({ responseTime, preprocessTime, mainprocessTime, postprocessTime })
            } else {
                const responseTime = 0
                const preprocessTime = 0
                const mainprocessTime = 0
                const postprocessTime = 0
                setPerformance({ responseTime, preprocessTime, mainprocessTime, postprocessTime })
            }
        }
    }, [voiceChangerClientRef.current])



    // (1-4) エラーステータス
    const ioErrorCountRef = useRef<number>(0)
    const resetIoErrorCount = () => {
        ioErrorCountRef.current = 0
        setIoErrorCount(ioErrorCountRef.current)
    }

    // (2-1) 初期化処理
    useEffect(() => {
        const initialized = async () => {
            if (!props.audioContext) {
                return
            }
            const voiceChangerClient = new VoiceChangerClient(props.audioContext, true, {
                notifySendBufferingTime: (val: number) => {
                    setBufferingTime(val)
                },
                notifyResponseTime: (val: number, perf?: number[]) => {
                    const responseTime = val
                    const preprocessTime = perf ? Math.ceil(perf[0] * 1000) : 0
                    const mainprocessTime = perf ? Math.ceil(perf[1] * 1000) : 0
                    const postprocessTime = perf ? Math.ceil(perf[2] * 1000) : 0
                    setPerformance({ responseTime, preprocessTime, mainprocessTime, postprocessTime })
                },
                notifyException: (mes: string) => {
                    if (mes.length > 0) {
                        console.log(`error:${mes}`)
                        ioErrorCountRef.current += 1
                        setIoErrorCount(ioErrorCountRef.current)
                    }
                },
                notifyVolume: (vol: number) => {
                    setVolume(vol)
                }
            })

            await voiceChangerClient.isInitialized()
            voiceChangerClientRef.current = voiceChangerClient
            setVoiceChangerClient(voiceChangerClientRef.current)
            console.log("[useClient] client initialized")

            // const audio = document.getElementById(props.audioOutputElementId) as HTMLAudioElement
            // audio.srcObject = voiceChangerClientRef.current.stream
            // audio.play()
            initializedResolveRef.current!()
            setInitialized(true)
        }
        initialized()
    }, [props.audioContext])

    const setAudioOutputElementId = (elemId: string) => {
        if (!voiceChangerClientRef.current) {
            console.warn("[voiceChangerClient] is not ready for set audio output.")
            return
        }
        const audio = document.getElementById(elemId) as HTMLAudioElement
        if (audio.paused) {
            audio.srcObject = voiceChangerClientRef.current.stream
            audio.play()
        }
    }

    // (2-2) 情報リロード
    const getInfo = useMemo(() => {
        return async () => {
            await initializedPromise
            await clientSetting.reloadClientSetting() // 実質的な処理の意味はない
            await serverSetting.reloadServerInfo()
        }
    }, [clientSetting.reloadClientSetting, serverSetting.reloadServerInfo])


    const clearSetting = async () => {
        // TBD
    }

    return {
        initialized,
        // 各種設定I/Fへの参照
        clientSetting,
        workletNodeSetting,
        workletSetting,
        serverSetting,
        indexedDBState,

        // モニタリングデータ
        bufferingTime,
        volume,
        performance,
        updatePerformance,

        // 情報取得
        getInfo,

        // 設定クリア
        clearSetting,

        // AudioOutputElement  設定
        setAudioOutputElementId,

        ioErrorCount,
        resetIoErrorCount
    }
}