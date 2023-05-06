import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"

export type IOBufferRowProps = {
}
export const IOBufferRow = (_props: IOBufferRowProps) => {
    const appState = useAppState()

    const ioBufferRow = useMemo(() => {
        if (appState.serverSetting.serverSetting.enableServerAudio == 0) {
            return <></>
        }
        const readBuf = appState.serverSetting.serverSetting.serverInputAudioBufferSize
        const writeBuf = appState.serverSetting.serverSetting.serverOutputAudioBufferSize

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">I/O Buffer</div>
                <div className="body-input-container">
                    <div className="left-padding-1">
                        In:
                        <select className="body-select" value={readBuf} onChange={(e) => {
                            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverInputAudioBufferSize: Number(e.target.value) })
                            appState.workletNodeSetting.trancateBuffer()
                        }}>
                            {
                                [1024 * 4, 1024 * 8, 1024 * 12, 1024 * 16, 1024 * 24, 1024 * 32].map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>

                    <div className="left-padding-1">
                        Out:
                        <select className="body-select" value={writeBuf} onChange={(e) => {
                            appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverOutputAudioBufferSize: Number(e.target.value) })
                            appState.workletNodeSetting.trancateBuffer()
                        }}>
                            {
                                [1024 * 4, 1024 * 8, 1024 * 12, 1024 * 16, 1024 * 24, 1024 * 32].map(x => {
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>

                    </div>
                </div>

                <div className="body-item-text"></div>
            </div>
        )
    }, [appState.serverSetting.serverSetting, appState.serverSetting.updateServerSettings])

    return ioBufferRow
}