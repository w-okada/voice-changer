import { Framework } from "@dannadori/voice-changer-client-js"
import React, { useEffect, useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"


export type MergeLabRowProps = {
}

type MergeElement = {
    filename: string
    strength: number
}

export const MergeLabRow = (_props: MergeLabRowProps) => {
    const [mergeElements, setMergeElements] = useState<MergeElement[]>([])
    const appState = useAppState()

    // スロットが変更されたときの初期化処理
    const newSlotChangeKey = useMemo(() => {
        return appState.serverSetting.serverSetting.modelSlots.reduce((prev, cur) => {
            return prev + "_" + cur.pyTorchModelFile
        }, "")
    }, [appState.serverSetting.serverSetting.modelSlots])

    useEffect(() => {
        // PyTorchモデルだけフィルタリング
        const models = appState.serverSetting.serverSetting.modelSlots.filter(x => { return x.pyTorchModelFile && x.pyTorchModelFile.length > 0 })
        if (models.length == 0) {
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


    const modelSwitchRow = useMemo(() => {
        const onMergeClicked = async () => {

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
                    {modelOptions.length == 0 ? <>no torch model or not same type</> : modelOptions}
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onMergeClicked}>merge</div>
                </div>
            </div>
        )
    }, [mergeElements, appState.serverSetting.serverSetting])

    return modelSwitchRow
}

