import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { ServerInfoSoVitsSVC } from "@dannadori/voice-changer-client-js";

export type DstIdRowProps = {
    showF0: boolean
    useServerInfo: boolean
    dstIdDisplayType: DstIdDisplayType
    staticIds: number[]
}

export const DstIdDisplayType = {
    "MMVCv13": "MMVCv13",
    "MMVCv15": "MMVCv15",
    "sovitsvc40": "sovitsvc40",
    "sovitsvc40v2": "sovitsvc40v2",
    "rvc": "rvc",
    "ddspsvc": "ddspsvc",
} as const
export type DstIdDisplayType = typeof DstIdDisplayType[keyof typeof DstIdDisplayType]

export const DstIdRow = (props: DstIdRowProps) => {
    const appState = useAppState()

    const dstIdRow = useMemo(() => {
        if (props.dstIdDisplayType != "MMVCv13") {
            return <></>
        }
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.dstId} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })

                    }}>
                        {
                            appState.clientSetting.clientSetting.speakers?.map(x => {
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
    }, [appState.serverSetting.serverSetting, appState.clientSetting.clientSetting.speakers, appState.serverSetting.updateServerSettings])

    const dstIdRowWithF0 = useMemo(() => {
        if (props.dstIdDisplayType != "MMVCv15") {
            return <></>
        }

        const selected = appState.clientSetting.clientSetting.correspondences?.find(x => {
            return x.sid == appState.serverSetting.serverSetting.dstId
        })
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.dstId} onChange={(e) => {
                        // const recF0 = calcDefaultF0Factor(appState.serverSetting.serverSetting.srcId, Number(e.target.value))
                        // appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value), f0Factor: recF0 })
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })

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

    const dstIdRowFromServer = useMemo(() => {
        if (props.dstIdDisplayType != "sovitsvc40" && props.dstIdDisplayType != "sovitsvc40v2") {
            return <></>
        }

        const settings = appState.serverSetting.serverSetting as ServerInfoSoVitsSVC
        const speakers = settings.speakers
        if (!speakers) {
            return <></>
        }

        const currentValue = Object.values(speakers).includes(appState.serverSetting.serverSetting.dstId) ? appState.serverSetting.serverSetting.dstId : -1

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={currentValue} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })

                    }}>
                        <option key="unknown" value={0}>default(0)</option>
                        {
                            Object.keys(speakers).map(x => {
                                return <option key={x} value={speakers[x]}>{x}({speakers[x]})</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings, appState.clientSetting.clientSetting.correspondences])

    const dstIdRowStatic = useMemo(() => {
        if (props.dstIdDisplayType != "rvc" && props.dstIdDisplayType != "ddspsvc") {
            return <></>
        }

        // const settings = appState.serverSetting.serverSetting as ServerInfoSoVitsSVC
        // const speakers = settings.speakers
        // if (!speakers) {
        //     return <></>
        // }

        const currentValue = Object.values(props.staticIds).includes(appState.serverSetting.serverSetting.dstId) ? appState.serverSetting.serverSetting.dstId : -1

        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={currentValue} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })

                    }}>
                        <option key="unknown" value={0}>default(0)</option>
                        {
                            Object.keys(props.staticIds).map(x => {
                                return <option key={x} value={x}>{x}</option>
                            })
                        }
                    </select>
                </div>
                <div className="body-item-text">
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])



    return (
        <>
            {dstIdRow}
            {dstIdRowWithF0}
            {dstIdRowFromServer}
            {dstIdRowStatic}
        </>
    )
}