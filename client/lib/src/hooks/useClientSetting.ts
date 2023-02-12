import { useState, useMemo, useRef, useEffect } from "react"
import { VoiceChangerClientSetting, Protocol, BufferSize, VoiceChangerMode, SampleRate, Speaker, DefaultVoiceChangerClientSetting, INDEXEDDB_KEY_CLIENT, Correspondence } from "../const"
import { createDummyMediaStream } from "../util"
import { VoiceChangerClient } from "../VoiceChangerClient"
import { useIndexedDB } from "./useIndexedDB"

export type UseClientSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    audioContext: AudioContext | null
}

export type ClientSettingState = {
    setting: VoiceChangerClientSetting;
    clearSetting: () => Promise<void>
    setServerUrl: (url: string) => void;
    setProtocol: (proto: Protocol) => void;
    setAudioInput: (audioInput: string | MediaStream | null) => Promise<void>
    setBufferSize: (bufferSize: BufferSize) => Promise<void>
    setVfForceDisabled: (vfForceDisabled: boolean) => Promise<void>
    setInputChunkNum: (num: number) => void;
    setVoiceChangerMode: (mode: VoiceChangerMode) => void
    setSampleRate: (num: SampleRate) => void
    setSpeakers: (speakers: Speaker[]) => void
    setCorrespondences: (file: File | null) => Promise<void>
    setInputGain: (val: number) => void
    setOutputGain: (val: number) => void
    start: () => Promise<void>
    stop: () => Promise<void>
    reloadClientSetting: () => Promise<void>
}

export const useClientSetting = (props: UseClientSettingProps): ClientSettingState => {
    const settingRef = useRef<VoiceChangerClientSetting>(DefaultVoiceChangerClientSetting)
    const [setting, _setSetting] = useState<VoiceChangerClientSetting>(settingRef.current)
    const { setItem, getItem, removeItem } = useIndexedDB()

    // 初期化 その１ DBから取得
    useEffect(() => {
        const loadCache = async () => {
            const setting = await getItem(INDEXEDDB_KEY_CLIENT)
            if (!setting) {
                // デフォルト設定
                console.log("No Chache",)
                const params = new URLSearchParams(location.search);
                const colab = params.get("colab")
                if (colab == "true") {
                    settingRef.current.protocol = "rest"
                    settingRef.current.inputChunkNum = 64
                } else {
                    settingRef.current.protocol = "sio"
                    settingRef.current.inputChunkNum = 32
                }
            } else {
                settingRef.current = setting as VoiceChangerClientSetting
            }
            _setSetting({ ...settingRef.current })
        }

        loadCache()
    }, [])
    // 初期化 その２ クライアントに設定
    useEffect(() => {
        if (!props.voiceChangerClient) return
        props.voiceChangerClient.setServerUrl(settingRef.current.mmvcServerUrl)
        props.voiceChangerClient.setInputChunkNum(settingRef.current.inputChunkNum)
        props.voiceChangerClient.setProtocol(settingRef.current.protocol)
        props.voiceChangerClient.setVoiceChangerMode(settingRef.current.voiceChangerMode)
        props.voiceChangerClient.setInputGain(settingRef.current.inputGain)

        // Input, bufferSize, VoiceFocus Disableは_setInputで設定
        _setInput()
    }, [props.voiceChangerClient])


    const setSetting = async (setting: VoiceChangerClientSetting) => {
        const storeData = { ...setting }
        if (typeof storeData.audioInput != "string") {
            storeData.audioInput = null
        }
        setItem(INDEXEDDB_KEY_CLIENT, storeData)
        _setSetting(setting)
    }

    const clearSetting = async () => {
        await removeItem(INDEXEDDB_KEY_CLIENT)
    }

    //////////////
    // 設定
    /////////////
    const setServerUrl = useMemo(() => {
        return (url: string) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setServerUrl(url, true)
            settingRef.current.mmvcServerUrl = url
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setProtocol = useMemo(() => {
        return (proto: Protocol) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setProtocol(proto)
            settingRef.current.protocol = proto
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const _setInput = async () => {
        if (!props.voiceChangerClient) return
        // console.log("[useClient] setup!(0)", settingRef.current.audioInput)
        if (!settingRef.current.audioInput || settingRef.current.audioInput == "none") {
            // console.log("[useClient] setup!(1)", settingRef.current.audioInput)
            const ms = createDummyMediaStream(props.audioContext!)
            await props.voiceChangerClient.setup(ms, settingRef.current.bufferSize, settingRef.current.forceVfDisable)

        } else {
            // console.log("[useClient] setup!(2)", settingRef.current.audioInput)
            await props.voiceChangerClient.setup(settingRef.current.audioInput, settingRef.current.bufferSize, settingRef.current.forceVfDisable)
        }
    }

    const setAudioInput = useMemo(() => {
        return async (audioInput: string | MediaStream | null) => {
            if (!props.voiceChangerClient) return
            settingRef.current.audioInput = audioInput
            await _setInput()
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setBufferSize = useMemo(() => {
        return async (bufferSize: BufferSize) => {
            if (!props.voiceChangerClient) return
            settingRef.current.bufferSize = bufferSize
            await _setInput()
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setVfForceDisabled = useMemo(() => {
        return async (vfForceDisabled: boolean) => {
            if (!props.voiceChangerClient) return
            settingRef.current.forceVfDisable = vfForceDisabled
            await _setInput()
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setInputChunkNum = useMemo(() => {
        return (num: number) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setInputChunkNum(num)
            settingRef.current.inputChunkNum = num
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setVoiceChangerMode = useMemo(() => {
        return (mode: VoiceChangerMode) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setVoiceChangerMode(mode)
            settingRef.current.voiceChangerMode = mode
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setSampleRate = useMemo(() => {
        return (num: SampleRate) => {
            if (!props.voiceChangerClient) return
            //props.voiceChangerClient.setSampleRate(num) // Not Implemented
            settingRef.current.sampleRate = num
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setSpeakers = useMemo(() => {
        return (speakers: Speaker[]) => {
            if (!props.voiceChangerClient) return
            settingRef.current.speakers = speakers
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setCorrespondences = useMemo(() => {
        return async (file: File | null) => {
            if (!props.voiceChangerClient) return
            if (!file) {
                settingRef.current.correspondences = []
            } else {
                const correspondenceText = await file.text()
                const cors = correspondenceText.split("\n").map(line => {
                    const items = line.split("|")
                    if (items.length != 3) {
                        console.warn("Invalid Correspondence Line:", line)
                        return null
                    } else {
                        const cor: Correspondence = {
                            sid: Number(items[0]),
                            correspondence: Number(items[1]),
                            dirname: items[2]
                        }
                        return cor
                    }
                }).filter(x => { return x != null }) as Correspondence[]
                settingRef.current.correspondences = cors
            }
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setInputGain = useMemo(() => {
        return (val: number) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setInputGain(val)
            settingRef.current.inputGain = val
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])
    const setOutputGain = useMemo(() => {
        return (val: number) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setOutputGain(val)
            settingRef.current.outputGain = val
            setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    //////////////
    // 操作
    /////////////
    // (1) start
    const start = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setServerUrl(setting.mmvcServerUrl, true)
            props.voiceChangerClient.start()
        }
    }, [setting.mmvcServerUrl, props.voiceChangerClient])
    // (2) stop
    const stop = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.stop()
        }
    }, [props.voiceChangerClient])
    const reloadClientSetting = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            await props.voiceChangerClient.getClientSettings()
        }
    }, [props.voiceChangerClient])

    return {
        setting,
        clearSetting,
        setServerUrl,
        setProtocol,
        setAudioInput,
        setBufferSize,
        setVfForceDisabled,
        setInputChunkNum,
        setVoiceChangerMode,
        setSampleRate,
        setSpeakers,
        setCorrespondences,
        setInputGain,
        setOutputGain,

        start,
        stop,
        reloadClientSetting
    }
}