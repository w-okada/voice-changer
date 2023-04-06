import React, { useMemo, useState } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const PerformanceRow = () => {
    const appState = useAppState()
    const [showPerformanceDetail, setShowPerformanceDetail] = useState<boolean>(false)

    const performanceRow = useMemo(() => {
        const performanceDetailLabel = showPerformanceDetail ? "[pre, main, post] <<" : "more >>"
        const performanceData = showPerformanceDetail ? `[${appState.performance.preprocessTime}, ${appState.performance.mainprocessTime},${appState.performance.postprocessTime}]` : ""
        return (
            <>
                <div className="body-row split-3-1-1-1-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1">monitor:</div>
                    <div className="body-item-text">vol<span className="body-item-text-small">(rms)</span></div>
                    <div className="body-item-text">buf<span className="body-item-text-small">(ms)</span></div>
                    <div className="body-item-text">res<span className="body-item-text-small">(ms)</span></div>
                    <div className="body-item-text">
                        <span onClick={() => { setShowPerformanceDetail(!showPerformanceDetail) }} >{performanceDetailLabel}</span>
                    </div>
                </div>
                <div className="body-row split-3-1-1-1-4 left-padding-1 guided">
                    <div className="body-item-title left-padding-1"></div>
                    <div className="body-item-text">{appState.volume.toFixed(4)}</div>
                    <div className="body-item-text">{appState.bufferingTime}</div>
                    <div className="body-item-text">{appState.performance.responseTime}</div>
                    <div className="body-item-text">{performanceData}</div>
                </div>
            </>
        )
    }, [appState.volume, appState.bufferingTime, appState.performance, showPerformanceDetail])

    return performanceRow
}