import React, { useMemo } from "react"

export const AnalyzerRow = () => {
    const analyzerRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1 ">Analyzer(Experimental)</div>
                <div className="body-button-container">
                </div>
            </div>
        )
    }, [])

    return analyzerRow
}