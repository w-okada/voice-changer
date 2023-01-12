import { BufferSize, createDummyMediaStream, DefaultVoiceChangerClientSetting, Protocol, SampleRate, Speaker, VoiceChangerClient, VoiceChangerClientSetting, VoiceChangerMode } from "@dannadori/voice-changer-client-js"
import { useState, useMemo, useRef, useEffect } from "react"

export type UseClientSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
    audioContext: AudioContext | null
}

export type ClientSettingState = {
    setting: VoiceChangerClientSetting;
    setServerUrl: (url: string) => void;
    setProtocol: (proto: Protocol) => void;
    setAudioInput: (audioInput: string | MediaStream | null) => Promise<void>
    setBufferSize: (bufferSize: BufferSize) => Promise<void>
    setVfForceDisabled: (vfForceDisabled: boolean) => Promise<void>
    setInputChunkNum: (num: number) => void;
    setVoiceChangerMode: (mode: VoiceChangerMode) => void
    setSampleRate: (num: SampleRate) => void
    setSpeakers: (speakers: Speaker[]) => void

    start: () => Promise<void>
    stop: () => Promise<void>
    reloadClientSetting: () => Promise<void>
}

export const useClientSetting = (props: UseClientSettingProps): ClientSettingState => {
    const settingRef = useRef<VoiceChangerClientSetting>(DefaultVoiceChangerClientSetting)
    const [setting, _setSetting] = useState<VoiceChangerClientSetting>(settingRef.current)

    //////////////
    // 設定
    /////////////
    const setServerUrl = useMemo(() => {
        return (url: string) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setServerUrl(url, true)
            settingRef.current.mmvcServerUrl = url
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setProtocol = useMemo(() => {
        return (proto: Protocol) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setProtocol(proto)
            settingRef.current.protocol = proto
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const _setInput = async () => {
        if (!props.voiceChangerClient) return

        if (!settingRef.current.audioInput || settingRef.current.audioInput == "none") {
            console.log("[useClient] setup!(1)", settingRef.current.audioInput)
            const ms = createDummyMediaStream(props.audioContext!)
            await props.voiceChangerClient.setup(ms, settingRef.current.bufferSize, settingRef.current.forceVfDisable)

        } else {
            console.log("[useClient] setup!(2)", settingRef.current.audioInput)
            await props.voiceChangerClient.setup(settingRef.current.audioInput, settingRef.current.bufferSize, settingRef.current.forceVfDisable)
        }
    }

    const setAudioInput = useMemo(() => {
        return async (audioInput: string | MediaStream | null) => {
            if (!props.voiceChangerClient) return
            settingRef.current.audioInput = audioInput
            await _setInput()
            _setSetting({ ...settingRef.current })
        }

    }, [props.voiceChangerClient])

    const setBufferSize = useMemo(() => {
        return async (bufferSize: BufferSize) => {
            if (!props.voiceChangerClient) return
            settingRef.current.bufferSize = bufferSize
            await _setInput()
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setVfForceDisabled = useMemo(() => {
        return async (vfForceDisabled: boolean) => {
            if (!props.voiceChangerClient) return
            settingRef.current.forceVfDisable = vfForceDisabled
            await _setInput()
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setInputChunkNum = useMemo(() => {
        return (num: number) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setInputChunkNum(num)
            settingRef.current.inputChunkNum = num
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setVoiceChangerMode = useMemo(() => {
        return (mode: VoiceChangerMode) => {
            if (!props.voiceChangerClient) return
            props.voiceChangerClient.setVoiceChangerMode(mode)
            settingRef.current.voiceChangerMode = mode
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setSampleRate = useMemo(() => {
        return (num: SampleRate) => {
            if (!props.voiceChangerClient) return
            //props.voiceChangerClient.setSampleRate(num) // Not Implemented
            settingRef.current.sampleRate = num
            _setSetting({ ...settingRef.current })
        }
    }, [props.voiceChangerClient])

    const setSpeakers = useMemo(() => {
        return (speakers: Speaker[]) => {
            if (!props.voiceChangerClient) return
            settingRef.current.speakers = speakers
            _setSetting({ ...settingRef.current })
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


    //////////////
    // Colab対応
    /////////////
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const colab = params.get("colab")
        if (colab == "true") {
            setProtocol("rest")
            setInputChunkNum(64)
        }
    }, [props.voiceChangerClient])


    return {
        setting,
        setServerUrl,
        setProtocol,
        setAudioInput,
        setBufferSize,
        setVfForceDisabled,
        setInputChunkNum,
        setVoiceChangerMode,
        setSampleRate,
        setSpeakers,

        start,
        stop,
        reloadClientSetting
    }
}