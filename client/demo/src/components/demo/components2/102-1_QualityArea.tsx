import React, { useMemo } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { F0Detector, } from "@dannadori/voice-changer-client-js"

export type QualityAreaProps = {
    detectors: string[]
}

export const QualityArea = (props: QualityAreaProps) => {
    const { clientSetting, serverSetting } = useAppState()
    const qualityArea = useMemo(() => {
        if (!serverSetting.updateServerSettings || !clientSetting.updateClientSetting || !serverSetting.serverSetting || !clientSetting.clientSetting) {
            return <></>
        }
        return (
            <div className="config-sub-area">
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">NOISE:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-noise-container">
                            <div className="config-sub-area-noise-checkbox-container">
                                <input type="checkbox" disabled={serverSetting.serverSetting.enableServerAudio != 0} checked={clientSetting.clientSetting.echoCancel} onChange={(e) => {
                                    clientSetting.updateClientSetting({ ...clientSetting.clientSetting, echoCancel: e.target.checked })
                                }} /> <span>Echo</span>

                            </div>
                            <div className="config-sub-area-noise-checkbox-container">
                                <input type="checkbox" disabled={serverSetting.serverSetting.enableServerAudio != 0} checked={clientSetting.clientSetting.noiseSuppression} onChange={(e) => {
                                    clientSetting.updateClientSetting({ ...clientSetting.clientSetting, noiseSuppression: e.target.checked })
                                }} /> <span>Sup1</span>

                            </div>
                            <div className="config-sub-area-noise-checkbox-container">
                                <input type="checkbox" disabled={serverSetting.serverSetting.enableServerAudio != 0} checked={clientSetting.clientSetting.noiseSuppression2} onChange={(e) => {
                                    clientSetting.updateClientSetting({ ...clientSetting.clientSetting, noiseSuppression2: e.target.checked })
                                }} />  <span>Sup2</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">F0 Det.:</div>
                    <div className="config-sub-area-control-field">
                        <select className="body-select" value={serverSetting.serverSetting.f0Detector} onChange={(e) => {
                            serverSetting.updateServerSettings({ ...serverSetting.serverSetting, f0Detector: e.target.value as F0Detector })
                        }}>
                            {
                                Object.values(props.detectors).map(x => {
                                    //@ts-ignore
                                    return <option key={x} value={x}>{x}</option>
                                })
                            }
                        </select>
                    </div>
                </div>
                <div className="config-sub-area-control">
                    <div className="config-sub-area-control-title">S.Thresh.:</div>
                    <div className="config-sub-area-control-field">
                        <div className="config-sub-area-slider-control">
                            <span className="config-sub-area-slider-control-kind"></span>
                            <span className="config-sub-area-slider-control-slider">
                                <input type="range" className="config-sub-area-slider-control-slider" min="0.00000" max="0.001" step="0.00001" value={serverSetting.serverSetting.silentThreshold || 0} onChange={(e) => {
                                    serverSetting.updateServerSettings({ ...serverSetting.serverSetting, silentThreshold: Number(e.target.value) })
                                }}></input>
                            </span>
                            <span className="config-sub-area-slider-control-val">{serverSetting.serverSetting.silentThreshold}</span>
                        </div>
                    </div>
                </div>

            </div>
        )
    }, [serverSetting.serverSetting, clientSetting.clientSetting, serverSetting.updateServerSettings, clientSetting.updateClientSetting])


    return qualityArea
}