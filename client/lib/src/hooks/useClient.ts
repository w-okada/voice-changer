import { useEffect, useMemo, useRef, useState } from "react"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { ClientSettingState, useClientSetting } from "./useClientSetting"
import { ServerSettingState, useServerSetting } from "./useServerSetting"
import { useWorkletSetting, WorkletSettingState } from "./useWorkletSetting"

export type UseClientProps = {
    audioContext: AudioContext | null
    audioOutputElementId: string
}

export type ClientState = {
    workletSetting: WorkletSettingState
    clientSetting: ClientSettingState
    serverSetting: ServerSettingState

    bufferingTime: number;
    responseTime: number;
    volume: number;

    getInfo: () => Promise<void>
}



export const useClient = (props: UseClientProps): ClientState => {

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


    // (1-2) 各種設定
    const clientSetting = useClientSetting({ voiceChangerClient, audioContext: props.audioContext })
    const workletSetting = useWorkletSetting({ voiceChangerClient })
    const serverSetting = useServerSetting({ voiceChangerClient })

    // (1-3) ステータス
    const [bufferingTime, setBufferingTime] = useState<number>(0)
    const [responseTime, setResponseTime] = useState<number>(0)
    const [volume, setVolume] = useState<number>(0)


    // (1-4) エラーステータス
    const errorCountRef = useRef<number>(0)


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
                notifyResponseTime: (val: number) => {
                    setResponseTime(val)
                },
                notifyException: (mes: string) => {
                    if (mes.length > 0) {
                        console.log(`error:${mes}`)
                        errorCountRef.current += 1
                        if (errorCountRef.current > 100) {
                            alert("エラーが頻発しています。対象としているフレームワークのモデルがロードされているか確認してください。")
                            errorCountRef.current = 0
                        }
                    }
                }
            }, {
                notifyVolume: (vol: number) => {
                    setVolume(vol)
                }
            })

            await voiceChangerClient.isInitialized()
            voiceChangerClientRef.current = voiceChangerClient
            setVoiceChangerClient(voiceChangerClientRef.current)
            console.log("[useClient] client initialized")

            const audio = document.getElementById(props.audioOutputElementId) as HTMLAudioElement
            audio.srcObject = voiceChangerClientRef.current.stream
            audio.play()
            initializedResolveRef.current!()
        }
        initialized()
    }, [props.audioContext])


    // (2-2) 情報リロード
    const getInfo = useMemo(() => {
        return async () => {
            await initializedPromise
            await clientSetting.reloadClientSetting()
            await serverSetting.reloadServerInfo()
        }
    }, [clientSetting, serverSetting])



    return {
        bufferingTime,
        responseTime,
        volume,

        getInfo,

        clientSetting,
        workletSetting,
        serverSetting,
    }
}