import { useEffect, useMemo, useRef, useState } from "react"
import { useAppSetting } from "../003_provider/AppSettingProvider"


export const AudioControllerStateType = {
    stop: "stop",
    record: "record",
    play: "play"
} as const
export type AudioControllerStateType = typeof AudioControllerStateType[keyof typeof AudioControllerStateType]

export type FixedUserData = {
    micWavBlob: Blob | undefined
    vfWavBlob: Blob | undefined
    micWavSamples: Float32Array | undefined
    vfWavSamples: Float32Array | undefined
    region: [number, number] | undefined
    micSpec: string
    vfSpec: string
}

type TempUserData = {
    micWavBlob: Blob | undefined
    vfWavBlob: Blob | undefined
    micWavSamples: Float32Array | undefined
    vfWavSamples: Float32Array | undefined
    region: [number, number] | undefined
    micSpec: string
    vfSpec: string
}

const initialFixedUserData: FixedUserData = {
    micWavBlob: undefined,
    vfWavBlob: undefined,
    micWavSamples: undefined,
    vfWavSamples: undefined,
    region: [0, 0],
    micSpec: "",
    vfSpec: ""
}

const initialTempUserData: TempUserData = {
    micWavBlob: undefined,
    vfWavBlob: undefined,
    micWavSamples: undefined,
    vfWavSamples: undefined,
    region: [0, 0],
    micSpec: "",
    vfSpec: ""
}

export type AudioControllerState = {
    audioControllerState: AudioControllerStateType
    unsavedRecord: boolean

    tempUserData: TempUserData
}
export type AudioControllerStateAndMethod = AudioControllerState & {
    setAudioControllerState: (val: AudioControllerStateType) => void
    setUnsavedRecord: (val: boolean) => void

    setTempWavBlob: (micBlob: Blob | undefined, vfBlob: Blob | undefined, micWavSamples: Float32Array | undefined, vfWavSamples: Float32Array | undefined, micSpec: string, vfSpec: string, region: [number, number]) => void
    saveWavBlob: () => void
    restoreFixedUserData: () => void
}

export const useAudioControllerState = (): AudioControllerStateAndMethod => {
    const { applicationSetting, appStateStorageState } = useAppSetting()
    const wavFilePrefix = useMemo(() => {
        const currentTextInfo = applicationSetting.applicationSetting.text.find(x => {
            return x.title == applicationSetting.applicationSetting.current_text
        })
        return currentTextInfo?.wavPrefix || "unknown"

    }, [applicationSetting.applicationSetting.current_text])

    const [audioControllerState, setAudioControllerState] = useState<AudioControllerStateType>("stop")
    const [unsavedRecord, setUnsavedRecord] = useState<boolean>(false);

    const fixedUserDataRef = useRef<FixedUserData>({ ...initialFixedUserData })
    const tempUserDataRef = useRef<TempUserData>({ ...initialTempUserData })

    const [_fixedUserData, setFixedUserData] = useState<FixedUserData>(fixedUserDataRef.current)
    const [tempUserData, setTempUserData] = useState<TempUserData>(tempUserDataRef.current)


    // 録音したデータをテンポラリバッファに格納
    const setTempWavBlob = (micBlob: Blob | undefined, vfBlob: Blob | undefined, micWavSamples: Float32Array | undefined, vfWavSamples: Float32Array | undefined, micSpec: string, vfSpec: string, region: [number, number]) => {
        tempUserDataRef.current.micWavBlob = micBlob
        tempUserDataRef.current.vfWavBlob = vfBlob
        tempUserDataRef.current.micWavSamples = micWavSamples
        tempUserDataRef.current.vfWavSamples = vfWavSamples
        tempUserDataRef.current.region = region
        tempUserDataRef.current.micSpec = micSpec
        tempUserDataRef.current.vfSpec = vfSpec
        setTempUserData({ ...tempUserDataRef.current })
    }

    // テンポラリをストレージにセーブ
    const saveWavBlob = () => {
        fixedUserDataRef.current.micWavBlob = tempUserDataRef.current.micWavBlob
        fixedUserDataRef.current.vfWavBlob = tempUserDataRef.current.vfWavBlob
        fixedUserDataRef.current.micWavSamples = tempUserDataRef.current.micWavSamples
        fixedUserDataRef.current.vfWavSamples = tempUserDataRef.current.vfWavSamples
        fixedUserDataRef.current.region = tempUserDataRef.current.region
        fixedUserDataRef.current.micSpec = tempUserDataRef.current.micSpec
        fixedUserDataRef.current.vfSpec = tempUserDataRef.current.vfSpec

        setFixedUserData({ ...fixedUserDataRef.current })
        appStateStorageState.saveUserData(applicationSetting.applicationSetting.current_text, wavFilePrefix, applicationSetting.applicationSetting.current_text_index, fixedUserDataRef.current)
    }

    // ストレージからロードして、確定データを復帰
    const loadWavBlob = async () => {
        const userData = await appStateStorageState.loadUserData(applicationSetting.applicationSetting.current_text, wavFilePrefix, applicationSetting.applicationSetting.current_text_index)
        if (!userData) {
            fixedUserDataRef.current.micWavBlob = undefined
            fixedUserDataRef.current.vfWavBlob = undefined
            fixedUserDataRef.current.micWavSamples = undefined
            fixedUserDataRef.current.vfWavSamples = undefined
            fixedUserDataRef.current.region = undefined
            fixedUserDataRef.current.micSpec = ""
            fixedUserDataRef.current.vfSpec = ""

        } else {
            fixedUserDataRef.current.micWavBlob = userData.micWavBlob
            fixedUserDataRef.current.vfWavBlob = userData.vfWavBlob
            fixedUserDataRef.current.micWavSamples = userData.micWavSamples
            fixedUserDataRef.current.vfWavSamples = userData.vfWavSamples
            fixedUserDataRef.current.region = userData.region
            fixedUserDataRef.current.micSpec = userData.micSpec
            fixedUserDataRef.current.vfSpec = userData.vfSpec
        }
        setFixedUserData({ ...fixedUserDataRef.current })
        restoreFixedUserData()
    }

    // テンポラリを捨てて確定データを復帰
    const restoreFixedUserData = () => {
        tempUserDataRef.current.micWavBlob = fixedUserDataRef.current.micWavBlob
        tempUserDataRef.current.vfWavBlob = fixedUserDataRef.current.vfWavBlob
        tempUserDataRef.current.micWavSamples = fixedUserDataRef.current.micWavSamples
        tempUserDataRef.current.vfWavSamples = fixedUserDataRef.current.vfWavSamples
        tempUserDataRef.current.region = fixedUserDataRef.current.region
        tempUserDataRef.current.micSpec = fixedUserDataRef.current.micSpec
        tempUserDataRef.current.vfSpec = fixedUserDataRef.current.vfSpec

        setTempUserData({ ...tempUserDataRef.current })
    }

    // ページ切り替え
    useEffect(() => {
        loadWavBlob()
    }, [applicationSetting.applicationSetting.current_text_index, wavFilePrefix])


    return {
        audioControllerState,
        unsavedRecord,
        tempUserData,

        setAudioControllerState,
        setUnsavedRecord,
        setTempWavBlob,
        saveWavBlob,
        restoreFixedUserData,
    }
}
