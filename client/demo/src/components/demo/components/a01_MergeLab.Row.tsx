import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { MergeElement } from "@dannadori/voice-changer-client-js"


export type MergeLabRowProps = {
}


export const MergeLabRow = (_props: MergeLabRowProps) => {
    const [mergeElements, setMergeElements] = useState<MergeElement[]>([])
    const appState = useAppState()
    const [defaultTune, setDefaultTune] = useState<number>(0)


    // スロットが変更されたときの初期化処理
    const newSlotChangeKey = useMemo(() => {
        if (!appState.serverSetting.serverSetting.modelSlots) {
            return ""
        }
        return appState.serverSetting.serverSetting.modelSlots.reduce((prev, cur) => {
            return prev + "_" + cur.modelFile
        }, "")
    }, [appState.serverSetting.serverSetting.modelSlots])


    // マージ用データセットの初期化
    const clearMergeModelSetting = useMemo(() => {
        return () => {
            // PyTorchモデルだけフィルタリング
            const models = appState.serverSetting.serverSetting.modelSlots.filter(x => { return x.modelFile && x.modelFile.endsWith("onnx") == false })
            if (models.length == 0) {
                setMergeElements([])
                return
            }

            const newMergeElements = models.map((x) => {
                const elem: MergeElement = { filename: x.modelFile, strength: 0 }
                return elem
            })
            setMergeElements(newMergeElements)
        }
    }, [appState.serverSetting.serverSetting.modelSlots])

    useEffect(() => {
        clearMergeModelSetting()
    }, [newSlotChangeKey])


    const mergeLabRow = useMemo(() => {
        const onMergeClicked = async () => {
            appState.serverSetting.mergeModel({
                command: "mix",
                defaultTune: defaultTune,
                defaultIndexRatio: 1,
                defaultProtect: 0.5,
                files: mergeElements
            })
        }

        const onMergeElementsChanged = (filename: string, strength: number) => {
            console.log("targetelement")
            const srcElements = mergeElements.filter(x => { return x.strength > 0 })
            const targetElement = mergeElements.find(x => { return x.filename == filename })
            if (!targetElement) {
                console.warn("target model is not found")
                return
            }
            // 一つ目の対象モデル
            if (srcElements.length == 0) {
                targetElement.strength = strength
                setMergeElements([...mergeElements])
                return
            }

            //二つ目以降

            const srcSample = appState.serverSetting.serverSetting.modelSlots.find(x => { return x.modelFile == srcElements[0].filename })
            const tgtSample = appState.serverSetting.serverSetting.modelSlots.find(x => { return x.modelFile == filename })
            if (!srcSample || !tgtSample) {
                console.warn("target model is not found", srcSample, tgtSample)
                return
            }
            if (
                srcSample.samplingRate != tgtSample.samplingRate ||
                srcSample.embChannels != tgtSample.embChannels ||
                srcSample.modelType != tgtSample.modelType
            ) {
                alert("current selected model is not same as the other selected.")
                console.log("current selected model is not same as the other selected.", srcSample.samplingRate, tgtSample.samplingRate,
                    srcSample.embChannels, tgtSample.embChannels,
                    srcSample.modelType, tgtSample.modelType)
                return
            }

            targetElement.strength = strength
            setMergeElements([...mergeElements])
            return
        }

        const modelOptions = mergeElements.map((x, index) => {
            let filename = ""
            if (x.filename.length > 0) {
                filename = x.filename.replace(/^.*[\\\/]/, '')
            } else {
                return (
                    <div key={index} >
                    </div>
                )
            }

            const modelInfo = appState.serverSetting.serverSetting.modelSlots.find(y => { return y.modelFile == x.filename })
            if (!modelInfo) {
                return (
                    <div key={index} >
                    </div>
                )
            }


            const f0str = modelInfo.f0 == true ? "f0" : "nof0"
            const srstr = Math.floor(modelInfo.samplingRate / 1000) + "K"
            const embedstr = modelInfo.embChannels
            const typestr = (() => {
                if (modelInfo.modelType == "pyTorchRVC" || modelInfo.modelType == "pyTorchRVCNono") {
                    return "org"
                } else if (modelInfo.modelType == "pyTorchRVCv2" || modelInfo.modelType == "pyTorchRVCv2Nono") {
                    return "g_v2"
                } else if (modelInfo.modelType == "pyTorchWebUI" || modelInfo.modelType == "pyTorchWebUINono") {
                    return "webui"
                } else {
                    return "unknown"
                }
            })()

            const metadata = `[${f0str},${srstr},${embedstr},${typestr}]`


            return (
                <div key={index} className="merge-field split-8-2">
                    <div className="merge-field-elem">{metadata} {filename}</div>
                    <div className="merge-field-elem">
                        <input type="range" className="body-item-input-slider" min="0" max="100" step="1" value={x.strength} onChange={(e) => {
                            onMergeElementsChanged(x.filename, Number(e.target.value))
                        }}></input>
                        <span className="body-item-input-slider-val">{x.strength}</span>
                    </div>
                </div >
            )
        })

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Model Merger</div>
                <div className="merge-field-container">
                    {modelOptions}

                    <div className="merge-field split-8-2">
                        <div className="merge-field-elem grey-bold">Default Tune</div>
                        <div className="merge-field-elem">
                            <input type="range" className="body-item-input-slider-2nd" min="-50" max="50" step="1" value={defaultTune} onChange={(e) => {
                                setDefaultTune(Number(e.target.value))
                            }}></input>
                            <span className="body-item-input-slider-val">{defaultTune}</span>
                        </div>
                    </div >


                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onMergeClicked}>merge</div>
                    <div className="body-button" onClick={clearMergeModelSetting}>clear</div>
                </div>
            </div>
        )
    }, [mergeElements, appState.serverSetting.serverSetting, defaultTune])

    return mergeLabRow

}