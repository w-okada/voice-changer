import { BufferSize } from "@dannadori/voice-changer-client-js"
import React, { useMemo, useState } from "react"
import { ClientState } from "./hooks/useClient"

export type UseConvertSettingProps = {
    clientState: ClientState
}

export type ConvertSettingState = {
    convertSetting: JSX.Element;
}

export const useConvertSetting = (props: UseConvertSettingProps): ConvertSettingState => {

    const bufferSizeRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Buffer Size</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.clientSetting.setting.bufferSize} onChange={(e) => {
                        props.clientState.clientSetting.setBufferSize(Number(e.target.value) as BufferSize)
                    }}>
                        {
                            Object.values(BufferSize).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.bufferSize, props.clientState.clientSetting.setBufferSize])

    const inputChunkNumRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Input Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.clientSetting.setting.inputChunkNum} onChange={(e) => {
                        props.clientState.clientSetting.setInputChunkNum(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.inputChunkNum, props.clientState.clientSetting.setInputChunkNum])

    const convertChunkNumRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Convert Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.serverSetting.setting.convertChunkNum} onChange={(e) => {
                        props.clientState.serverSetting.setConvertChunkNum(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.convertChunkNum, props.clientState.serverSetting.setConvertChunkNum])

    const gpuRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">GPU</div>
                <div className="body-input-container">
                    <input type="number" min={-2} max={5} step={1} value={props.clientState.serverSetting.setting.gpu} onChange={(e) => {
                        props.clientState.serverSetting.setGpu(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.gpu, props.clientState.serverSetting.setGpu])

    const crossFadeOverlapRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0.1} max={1} step={0.1} value={props.clientState.serverSetting.setting.crossFadeOverlapRate} onChange={(e) => {
                        props.clientState.serverSetting.setCrossFadeOverlapRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.crossFadeOverlapRate, props.clientState.serverSetting.setCrossFadeOverlapRate])

    const crossFadeOffsetRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={props.clientState.serverSetting.setting.crossFadeOffsetRate} onChange={(e) => {
                        props.clientState.serverSetting.setCrossFadeOffsetRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.crossFadeOffsetRate, props.clientState.serverSetting.setCrossFadeOffsetRate])

    const crossFadeEndRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Cross Fade End Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={props.clientState.serverSetting.setting.crossFadeEndRate} onChange={(e) => {
                        props.clientState.serverSetting.setCrossFadeEndRate(Number(e.target.value))
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.crossFadeEndRate, props.clientState.serverSetting.setCrossFadeEndRate])

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
                {crossFadeOverlapRateRow}
                {crossFadeOffsetRateRow}
                {crossFadeEndRateRow}
            </>
        )
    }, [bufferSizeRow, inputChunkNumRow, convertChunkNumRow, gpuRow, crossFadeOverlapRateRow, crossFadeOffsetRateRow, crossFadeEndRateRow])

    return {
        convertSetting,
    }

}


