import { DefaultVoiceChangerRequestParamas, DefaultVoiceChangerOptions, BufferSize } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"

export type SpeakerSettingState = {
    convertSetting: JSX.Element;
    bufferSize: BufferSize;
    inputChunkNum: number;
    convertChunkNum: number;
    gpu: number;
    crossFadeOffsetRate: number;
    crossFadeEndRate: number;
}

export const useConvertSetting = (): SpeakerSettingState => {

    const [bufferSize, setBufferSize] = useState<BufferSize>(1024)
    const [inputChunkNum, setInputChunkNum] = useState<number>(DefaultVoiceChangerOptions.inputChunkNum)
    const [convertChunkNum, setConvertChunkNum] = useState<number>(DefaultVoiceChangerRequestParamas.convertChunkNum)
    const [gpu, setGpu] = useState<number>(DefaultVoiceChangerRequestParamas.gpu)

    const [crossFadeOffsetRate, setCrossFadeOffsetRate] = useState<number>(DefaultVoiceChangerRequestParamas.crossFadeOffsetRate)
    const [crossFadeEndRate, setCrossFadeEndRate] = useState<number>(DefaultVoiceChangerRequestParamas.crossFadeEndRate)

    const bufferSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Buffer Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={bufferSize} onChange={(e) => { setBufferSize(Number(e.target.value) as BufferSize) }}>
                        {
                            Object.values(BufferSize).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [bufferSize])

    const inputChunkNumRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Input Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={inputChunkNum} onChange={(e) => { setInputChunkNum(Number(e.target.value)) }} />
                </div>
            </div>
        )
    }, [inputChunkNum])

    const convertChunkNumRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Convert Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={convertChunkNum} onChange={(e) => { setConvertChunkNum(Number(e.target.value)) }} />
                </div>
            </div>
        )
    }, [convertChunkNum])

    const gpuRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">GPU</div>
                <div className="body-input-container">
                    <input type="number" min={-2} max={5} step={1} value={gpu} onChange={(e) => { setGpu(Number(e.target.value)) }} />
                </div>
            </div>
        )
    }, [gpu])

    const crossFadeOffsetRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={crossFadeOffsetRate} onChange={(e) => { setCrossFadeOffsetRate(Number(e.target.value)) }} />
                </div>
            </div>
        )
    }, [crossFadeOffsetRate])

    const crossFadeEndRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Cross Fade End Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={crossFadeEndRate} onChange={(e) => { setCrossFadeEndRate(Number(e.target.value)) }} />
                </div>
            </div>
        )
    }, [crossFadeEndRate])

    const convertSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Converter Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {bufferSizeRow}
                {inputChunkNumRow}
                {convertChunkNumRow}
                {gpuRow}
                {crossFadeOffsetRateRow}
                {crossFadeEndRateRow}
            </>
        )
    }, [bufferSizeRow, inputChunkNumRow, convertChunkNumRow, gpuRow, crossFadeOffsetRateRow, crossFadeEndRateRow])

    return {
        convertSetting,
        bufferSize,
        inputChunkNum,
        convertChunkNum,
        gpu,
        crossFadeOffsetRate,
        crossFadeEndRate,
    }

}


