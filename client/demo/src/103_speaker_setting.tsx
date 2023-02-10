import React, { useMemo, useState } from "react"
import { ClientState } from "@dannadori/voice-changer-client-js";

export type UseSpeakerSettingProps = {
    clientState: ClientState
}

export const useSpeakerSetting = (props: UseSpeakerSettingProps) => {
    const [editSpeakerTargetId, setEditSpeakerTargetId] = useState<number>(0)
    const [editSpeakerTargetName, setEditSpeakerTargetName] = useState<string>("")

    const srcIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.serverSetting.setting.srcId} onChange={(e) => {
                        props.clientState.serverSetting.setSrcId(Number(e.target.value))
                    }}>
                        {
                            props.clientState.clientSetting.setting.speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.speakers, props.clientState.serverSetting.setting.srcId, props.clientState.serverSetting.setSrcId])

    const dstIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.serverSetting.setting.dstId} onChange={(e) => {
                        props.clientState.serverSetting.setDstId(Number(e.target.value))
                    }}>
                        {
                            props.clientState.clientSetting.setting.speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.speakers, props.clientState.serverSetting.setting.dstId, props.clientState.serverSetting.setDstId])

    const editSpeakerIdMappingRow = useMemo(() => {
        const onSetSpeakerMappingClicked = async () => {
            const targetId = editSpeakerTargetId
            const targetName = editSpeakerTargetName
            const targetSpeaker = props.clientState.clientSetting.setting.speakers.find(x => { return x.id == targetId })
            if (targetSpeaker) {
                if (targetName.length == 0) { // Delete
                    const newSpeakers = props.clientState.clientSetting.setting.speakers.filter(x => { return x.id != targetId })
                    props.clientState.clientSetting.setSpeakers(newSpeakers)
                } else { // Update
                    targetSpeaker.name = targetName
                    props.clientState.clientSetting.setSpeakers([...props.clientState.clientSetting.setting.speakers])
                }
            } else {
                if (targetName.length == 0) { // Noop
                } else {// add
                    props.clientState.clientSetting.setting.speakers.push({
                        id: targetId,
                        name: targetName
                    })
                    props.clientState.clientSetting.setSpeakers([...props.clientState.clientSetting.setting.speakers])
                }
            }
        }
        return (
            <div className="body-row split-3-1-2-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Edit Speaker Mapping</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={editSpeakerTargetId} onChange={(e) => {
                        const id = Number(e.target.value)
                        setEditSpeakerTargetId(id)
                        setEditSpeakerTargetName(props.clientState.clientSetting.setting.speakers.find(x => { return x.id == id })?.name || "")
                    }} />
                </div>
                <div className="body-input-container">
                    <input type="text" value={editSpeakerTargetName} onChange={(e) => {
                        setEditSpeakerTargetName(e.target.value)
                    }} />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetSpeakerMappingClicked}>set</div>
                </div>
            </div>
        )
    }, [props.clientState.clientSetting.setting.speakers, editSpeakerTargetId, editSpeakerTargetName])


    const f0FactorRow = useMemo(() => {
        return (
            <div className="body-row split-3-2-1-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">F0 Factor</div>
                <div className="body-input-container">
                    <input type="range" className="body-item-input" min="0.1" max="5.0" step="0.1" value={props.clientState.serverSetting.setting.f0Factor} onChange={(e) => {
                        props.clientState.serverSetting.setF0Factor(Number(e.target.value))
                    }}></input>
                </div>
                <div className="body-item-text">
                    <div>{props.clientState.serverSetting.setting.f0Factor}</div>
                </div>
                <div className="body-item-text"></div>
            </div>
        )
    }, [props.clientState.serverSetting.setting.f0Factor, props.clientState.serverSetting.setF0Factor])

    const speakerSetting = useMemo(() => {
        return (
            <>
                <div className="body-row split-3-7 left-padding-1">
                    <div className="body-sub-section-title">Speaker Setting</div>
                    <div className="body-select-container">
                    </div>
                </div>
                {srcIdRow}
                {dstIdRow}
                {editSpeakerIdMappingRow}
                {f0FactorRow}
            </>
        )
    }, [srcIdRow, dstIdRow, editSpeakerIdMappingRow, f0FactorRow])

    return {
        speakerSetting,
    }

}


