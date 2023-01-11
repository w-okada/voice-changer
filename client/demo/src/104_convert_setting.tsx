import { DefaultVoiceChangerRequestParamas, DefaultVoiceChangerOptions, BufferSize } from "@dannadori/voice-changer-client-js"
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
                    <select className="body-select" value={props.clientState.settingState.bufferSize} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            bufferSize: Number(e.target.value) as BufferSize
                        })
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
    }, [props.clientState.settingState])

    const inputChunkNumRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Input Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.settingState.inputChunkNum} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            inputChunkNum: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const convertChunkNumRow = useMemo(() => {
        return (

            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Convert Chunk Num(128sample/chunk)</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.settingState.convertChunkNum} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            convertChunkNum: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const gpuRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">GPU</div>
                <div className="body-input-container">
                    <input type="number" min={-2} max={5} step={1} value={props.clientState.settingState.gpu} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            gpu: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const crossFadeOverlapRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Overlap Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0.1} max={1} step={0.1} value={props.clientState.settingState.crossFadeOverlapRate} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            crossFadeOverlapRate: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const crossFadeOffsetRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title  left-padding-1">Cross Fade Offset Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={props.clientState.settingState.crossFadeOffsetRate} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            crossFadeOffsetRate: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const crossFadeEndRateRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Cross Fade End Rate</div>
                <div className="body-input-container">
                    <input type="number" min={0} max={1} step={0.1} value={props.clientState.settingState.crossFadeEndRate} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            crossFadeEndRate: Number(e.target.value)
                        })
                    }} />
                </div>
            </div>
        )
    }, [props.clientState.settingState])

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


