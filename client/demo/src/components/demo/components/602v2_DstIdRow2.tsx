import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { ServerInfoSoVitsSVC } from "@dannadori/voice-changer-client-js";

export type DstIdRow2Props = {
    selectableIds: (number | string)[]
}

export const DstIdRow2 = (props: DstIdRow2Props) => {
    const appState = useAppState()

    const dstIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={appState.serverSetting.serverSetting.dstId} onChange={(e) => {
                        appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, dstId: Number(e.target.value) })
                    }}>
                        {
                            props.selectableIds.map(x => {
                                return <option key={x} value={x}>{x}({x})</option>
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
        </>
    )
}