import React, { useMemo } from "react"
import { SamplingRow } from "./510-1_SamplingRow"
import { SamplingPlayRow } from "./510-2_SamplingPlayRow"

export type AnalyzerRowProps = {
}

export const AnalyzerRow = (_props: AnalyzerRowProps) => {
    const analyzerRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Analyzer</div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [])

    return (
        <>
            {analyzerRow}
            <SamplingRow />
            <SamplingPlayRow />
        </>
    )

}