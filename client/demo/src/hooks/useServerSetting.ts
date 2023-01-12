import { DefaultVoiceChangerServerSetting, Framework, OnnxExecutionProvider, ServerInfo, ServerSettingKey, VoiceChangerClient, VoiceChangerServerSetting, } from "@dannadori/voice-changer-client-js"
import { useState, useMemo, useRef, useEffect } from "react"


export type FileUploadSetting = {
    pyTorchModel: File | null
    configFile: File | null
    onnxModel: File | null
}
const InitialFileUploadSetting: FileUploadSetting = {
    pyTorchModel: null,
    configFile: null,
    onnxModel: null,
}
export type UseServerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type ServerSettingState = {
    setting: VoiceChangerServerSetting;
    serverInfo: ServerInfo | undefined;
    fileUploadSetting: FileUploadSetting
    setFramework: (framework: Framework) => Promise<boolean>;
    setOnnxExecutionProvider: (provider: OnnxExecutionProvider) => Promise<boolean>;
    setSrcId: (num: number) => Promise<boolean>;
    setDstId: (num: number) => Promise<boolean>;
    setConvertChunkNum: (num: number) => Promise<boolean>;
    setMinConvertSize: (num: number) => Promise<boolean>
    setGpu: (num: number) => Promise<boolean>;
    setCrossFadeOffsetRate: (num: number) => Promise<boolean>;
    setCrossFadeEndRate: (num: number) => Promise<boolean>;
    setCrossFadeOverlapRate: (num: number) => Promise<boolean>;
    reloadServerInfo: () => Promise<void>;
    setFileUploadSetting: (val: FileUploadSetting) => void
    loadModel: () => Promise<void>
    uploadProgress: number
    isUploading: boolean
}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    const settingRef = useRef<VoiceChangerServerSetting>(DefaultVoiceChangerServerSetting)
    const [setting, _setSetting] = useState<VoiceChangerServerSetting>(settingRef.current)
    const [serverInfo, _setServerInfo] = useState<ServerInfo>()
    const [fileUploadSetting, setFileUploadSetting] = useState<FileUploadSetting>(InitialFileUploadSetting)

    //////////////
    // 設定
    /////////////
    //// サーバに設定後、反映された情報と照合して値が一致していることを確認。一致していない場合はalert
    const _set_and_store = async (key: ServerSettingKey, newVal: string) => {
        if (!props.voiceChangerClient) return false

        const res = await props.voiceChangerClient.updateServerSettings(key, "" + newVal)

        _setServerInfo(res)
        if (newVal == res[key]) {
            _setSetting({
                ...settingRef.current,
                convertChunkNum: res.convertChunkNum,
                minConvertSize: res.minConvertSize,
                srcId: res.srcId,
                dstId: res.dstId,
                gpu: res.gpu,
                crossFadeOffsetRate: res.crossFadeOffsetRate,
                crossFadeEndRate: res.crossFadeEndRate,
                crossFadeOverlapRate: res.crossFadeOverlapRate,
                framework: res.framework,
                onnxExecutionProvider: (!!res.onnxExecutionProvider && res.onnxExecutionProvider.length > 0) ? res.onnxExecutionProvider[0] as OnnxExecutionProvider : DefaultVoiceChangerServerSetting.onnxExecutionProvider
            })
            return true
        } else {
            alert(`[ServerSetting] setting failed. [key:${key}, new:${newVal}, res:${res[key]}]`)
            return false
        }

    }

    const setFramework = useMemo(() => {
        return async (framework: Framework) => {
            return await _set_and_store(ServerSettingKey.framework, "" + framework)
        }
    }, [props.voiceChangerClient])

    const setOnnxExecutionProvider = useMemo(() => {
        return async (provider: OnnxExecutionProvider) => {
            return await _set_and_store(ServerSettingKey.onnxExecutionProvider, "" + provider)
        }
    }, [props.voiceChangerClient])

    const setSrcId = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.srcId, "" + num)
        }
    }, [props.voiceChangerClient])

    const setDstId = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.dstId, "" + num)
        }
    }, [props.voiceChangerClient])

    const setConvertChunkNum = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.convertChunkNum, "" + num)
        }
    }, [props.voiceChangerClient])

    const setMinConvertSize = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.minConvertSize, "" + num)
        }
    }, [props.voiceChangerClient])


    const setGpu = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.gpu, "" + num)
        }
    }, [props.voiceChangerClient])

    const setCrossFadeOffsetRate = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.crossFadeOffsetRate, "" + num)
        }
    }, [props.voiceChangerClient])
    const setCrossFadeEndRate = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.crossFadeEndRate, "" + num)
        }
    }, [props.voiceChangerClient])
    const setCrossFadeOverlapRate = useMemo(() => {
        return async (num: number) => {
            return await _set_and_store(ServerSettingKey.crossFadeOverlapRate, "" + num)
        }
    }, [props.voiceChangerClient])



    //////////////
    // 操作
    /////////////
    const [uploadProgress, setUploadProgress] = useState<number>(0)
    const [isUploading, setIsUploading] = useState<boolean>(false)

    // (e) モデルアップロード
    const _uploadFile = useMemo(() => {
        return async (file: File, onprogress: (progress: number, end: boolean) => void) => {
            if (!props.voiceChangerClient) return
            const num = await props.voiceChangerClient.uploadFile(file, onprogress)
            const res = await props.voiceChangerClient.concatUploadedFile(file, num)
            console.log("uploaded", num, res)
        }
    }, [props.voiceChangerClient])
    const loadModel = useMemo(() => {
        return async () => {
            if (!fileUploadSetting.pyTorchModel && !fileUploadSetting.onnxModel) {
                alert("PyTorchモデルとONNXモデルのどちらか一つ以上指定する必要があります。")
                return
            }
            if (!fileUploadSetting.configFile) {
                alert("Configファイルを指定する必要があります。")
                return
            }
            if (!props.voiceChangerClient) return
            setUploadProgress(0)
            setIsUploading(true)
            const models = [fileUploadSetting.pyTorchModel, fileUploadSetting.onnxModel].filter(x => { return x != null }) as File[]
            for (let i = 0; i < models.length; i++) {
                const progRate = 1 / models.length
                const progOffset = 100 * i * progRate
                await _uploadFile(models[i], (progress: number, end: boolean) => {
                    // console.log(progress * progRate + progOffset, end, progRate,)
                    setUploadProgress(progress * progRate + progOffset)
                })
            }

            await _uploadFile(fileUploadSetting.configFile, (progress: number, end: boolean) => {
                console.log(progress, end)
            })

            const serverInfo = await props.voiceChangerClient.loadModel(fileUploadSetting.configFile, fileUploadSetting.pyTorchModel, fileUploadSetting.onnxModel)
            console.log(serverInfo)
            setUploadProgress(0)
            setIsUploading(false)
        }
    }, [fileUploadSetting, props.voiceChangerClient])

    const reloadServerInfo = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return
            const res = await props.voiceChangerClient.getServerSettings()
            _setServerInfo(res)
            _setSetting({
                ...settingRef.current,
                convertChunkNum: res.convertChunkNum,
                srcId: res.srcId,
                dstId: res.dstId,
                gpu: res.gpu,
                crossFadeOffsetRate: res.crossFadeOffsetRate,
                crossFadeEndRate: res.crossFadeEndRate,
                crossFadeOverlapRate: res.crossFadeOverlapRate,
                framework: res.framework,
                onnxExecutionProvider: (!!res.onnxExecutionProvider && res.onnxExecutionProvider.length > 0) ? res.onnxExecutionProvider[0] as OnnxExecutionProvider : DefaultVoiceChangerServerSetting.onnxExecutionProvider
            })

        }
    }, [props.voiceChangerClient])


    //////////////
    // デフォルト設定
    /////////////
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const colab = params.get("colab")
        if (colab == "true") {
        } else {
        }
    }, [props.voiceChangerClient])



    return {
        setting,
        serverInfo,
        fileUploadSetting,
        setFramework,
        setOnnxExecutionProvider,
        setSrcId,
        setDstId,
        setConvertChunkNum,
        setMinConvertSize,
        setGpu,
        setCrossFadeOffsetRate,
        setCrossFadeEndRate,
        setCrossFadeOverlapRate,
        reloadServerInfo,
        setFileUploadSetting,
        loadModel,
        uploadProgress,
        isUploading,
    }
}