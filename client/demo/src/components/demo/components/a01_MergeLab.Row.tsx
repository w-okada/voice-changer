import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { MergeElement } from "@dannadori/voice-changer-client-js"


export type MergeLabRowProps = {
}


export const MergeLabRow = (_props: MergeLabRowProps) => {
    const [mergeElements, setMergeElements] = useState<MergeElement[]>([])
    const appState = useAppState()
    const [defaultTrans, setDefaultTrans] = useState<number>(0)

    // スロットが変更されたときの初期化処理
    const newSlotChangeKey = useMemo(() => {
        console.log("appState.serverSetting.serverSetting.modelSlots", appState.serverSetting.serverSetting.modelSlots)
        return appState.serverSetting.serverSetting.modelSlots.reduce((prev, cur) => {
            return prev + "_" + cur.pyTorchModelFile
        }, "")
    }, [appState.serverSetting.serverSetting.modelSlots])

    console.log("newSlotChangeKey", newSlotChangeKey)
    useEffect(() => {
        // PyTorchモデルだけフィルタリング
        const models = appState.serverSetting.serverSetting.modelSlots.filter(x => { return x.pyTorchModelFile && x.pyTorchModelFile.length > 0 })
        if (models.length == 0) {
            setMergeElements([])
            return
        }

        // サンプリングレート、埋め込み次元数、モデルタイプが同一の場合のみ処理可能

        if (
            models.map(x => { return x.samplingRate }).every((x, _i, arr) => x == arr[0]) &&
            models.map(x => { return x.embChannels }).every((x, _i, arr) => x == arr[0]) &&
            models.map(x => { return x.modelType }).every((x, _i, arr) => x == arr[0])
        ) {

            const newMergeElements = models.map((x) => {
                const elem: MergeElement = { filename: x.pyTorchModelFile, strength: 100 }
                return elem
            })
            setMergeElements(newMergeElements)
        } else {
            console.log("not all model is same properties.")
            setMergeElements([])
        }
    }, [newSlotChangeKey])


    const mergeLabRow = useMemo(() => {
        const onMergeClicked = async () => {
            appState.serverSetting.mergeModel({
                command: "mix",
                defaultTrans: defaultTrans,
                files: mergeElements
            })
        }
        const onMergeElementsChanged = (filename: string, strength: number) => {
            const newMergeElements = mergeElements.map(x => {
                if (x.filename != filename) return x

                x.strength = strength
                return x
            })
            setMergeElements(newMergeElements)
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

            return (
                <div key={index} className="merge-field">
                    <div className="merge-field-elem">{filename}</div>
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
                    {modelOptions.length == 0 ? <>not PyTorch model or not same type</> : modelOptions}

                    <div className="merge-field">
                        <div className="merge-field-elem grey-bold">Default Tune</div>
                        <div className="merge-field-elem">
                            <input type="range" className="body-item-input-slider-2nd" min="-50" max="50" step="1" value={defaultTrans} onChange={(e) => {
                                setDefaultTrans(Number(e.target.value))
                            }}></input>
                            <span className="body-item-input-slider-val">{defaultTrans}</span>
                        </div>
                    </div >


                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onMergeClicked}>merge</div>
                </div>
            </div>
        )
    }, [mergeElements, appState.serverSetting.serverSetting, defaultTrans])

    return mergeLabRow

}