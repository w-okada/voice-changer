import React, { useMemo } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const SrcIdRow = () => {
    const appState = useAppState()
    const speakerSetting = appState.appGuiSettingState.appGuiSetting.front.speakerSetting

    const srcIdRow = useMemo(() => {
        if (!speakerSetting.srcIdEnable) {
            return <></>
        }
        if (speakerSetting.showSpeakerF0) {
            return <></>
        }
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.srcId} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, srcId: Number(e.target.value) })
                    }}>
                        {
                            appState.clientSetting.clientSetting.speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })

                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, appState.clientSetting.clientSetting.speakers])

    const srcIdRowWithF0 = useMemo(() => {
        if (!speakerSetting.srcIdEnable) {
            return <></>
        }
        if (!speakerSetting.showSpeakerF0) {
            return <></>
        }
        const selected = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.srcId
        })
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.srcId} onChange={(e) => {
                        // const recF0 = calcDefaultF0Factor(Number(e.target.value), appState.serverSetting.serverSetting.dstId)
                        // appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, srcId: Number(e.target.value), f0Factor: recF0 })
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, srcId: Number(e.target.value) })
                    }}>
                        {
                            appState.clientSetting.clientSetting.correspondences?.map(x => {
                                return <option key={x.sid} value={x.sid}>{x.dirname}({x.sid})</option>
                            })

                        }
                    </select>
                </div>
                <div className="body-item-text">
                    <div>F0: {selected?.correspondence.toFixed(1) || ""}</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, appState.clientSetting.clientSetting.correspondences])

    return (
        <>
            {srcIdRow}
            {srcIdRowWithF0}
        </>
    )
}