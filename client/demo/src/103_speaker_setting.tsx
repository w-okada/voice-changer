import React, { useMemo } from "react"
import { ClientState } from "./hooks/useClient"

export type UseSpeakerSettingProps = {
    clientState: ClientState
}

export const useSpeakerSetting = (props: UseSpeakerSettingProps) => {

    const srcIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Source Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.settingState.srcId} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            srcId: Number(e.target.value)
                        })
                    }}>
                        {
                            props.clientState.settingState.speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const dstIdRow = useMemo(() => {
        return (
            <div className="body-row split-3-7 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Destination Speaker Id</div>
                <div className="body-select-container">
                    <select className="body-select" value={props.clientState.settingState.dstId} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            dstId: Number(e.target.value)
                        })
                    }}>
                        {
                            props.clientState.settingState.speakers.map(x => {
                                return <option key={x.id} value={x.id}>{x.name}({x.id})</option>
                            })
                        }
                    </select>
                </div>
            </div>
        )
    }, [props.clientState.settingState])

    const editSpeakerIdMappingRow = useMemo(() => {
        const onSetSpeakerMappingClicked = async () => {
            const targetId = props.clientState.settingState.editSpeakerTargetId
            const targetName = props.clientState.settingState.editSpeakerTargetName
            const targetSpeaker = props.clientState.settingState.speakers.find(x => { return x.id == targetId })
            if (targetSpeaker) {
                if (targetName.length == 0) { // Delete
                    const newSpeakers = props.clientState.settingState.speakers.filter(x => { return x.id != targetId })
                    props.clientState.setSettingState({
                        ...props.clientState.settingState,
                        speakers: newSpeakers
                    })
                } else { // Update
                    targetSpeaker.name = targetName
                    props.clientState.setSettingState({
                        ...props.clientState.settingState,
                        speakers: props.clientState.settingState.speakers
                    })
                }
            } else {
                if (targetName.length == 0) { // Noop
                } else {// add
                    props.clientState.settingState.speakers.push({
                        id: targetId,
                        name: targetName
                    })
                    props.clientState.setSettingState({
                        ...props.clientState.settingState,
                        speakers: props.clientState.settingState.speakers
                    })
                }
            }
        }
        return (
            <div className="body-row split-3-1-2-4 left-padding-1 guided">
                <div className="body-item-title left-padding-1">Edit Speaker Mapping</div>
                <div className="body-input-container">
                    <input type="number" min={1} max={256} step={1} value={props.clientState.settingState.editSpeakerTargetId} onChange={(e) => {
                        const id = Number(e.target.value)
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            editSpeakerTargetId: id,
                            editSpeakerTargetName: props.clientState.settingState.speakers.find(x => { return x.id == id })?.name || ""
                        })
                    }} />
                </div>
                <div className="body-input-container">
                    <input type="text" value={props.clientState.settingState.editSpeakerTargetName} onChange={(e) => {
                        props.clientState.setSettingState({
                            ...props.clientState.settingState,
                            editSpeakerTargetName: e.target.value
                        })
                    }} />
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onSetSpeakerMappingClicked}>set</div>
                </div>
            </div>
        )
    }, [props.clientState.settingState])


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
            </>
        )
    }, [srcIdRow, dstIdRow, editSpeakerIdMappingRow])

    return {
        speakerSetting,
    }

}


