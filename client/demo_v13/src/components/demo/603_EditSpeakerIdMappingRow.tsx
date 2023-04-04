import React, { useMemo, useState } from "react"
import { useAppState } from "../../001_provider/001_AppStateProvider"

export const EditSpeakerIdMappingRow = () => {
    const appState = useAppState()
    const speakerSetting = appState.appGuiSettingState.appGuiSetting.front.speakerSetting
    const [editSpeakerTargetId, setEditSpeakerTargetId] = useState<number>(0)
    const [editSpeakerTargetName, setEditSpeakerTargetName] = useState<string>("")

    const editSpeakerIdMappingRow = useMemo(() => {
        if (!speakerSetting.editSpeakerIdMappingEnable) {
            return <></>
        }

        const onSetSpeakerMappingClicked = async () => {
            const targetId = editSpeakerTargetId
            const targetName = editSpeakerTargetName
            const targetSpeaker = appState.clientSetting.clientSetting.speakers.find(x => { return x.id == targetId })
            if (targetSpeaker) {
                if (targetName.length == 0) { // Delete
                    const newSpeakers = appState.clientSetting.clientSetting.speakers.filter(x => { return x.id != targetId })
                    appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, speakers: newSpeakers })
                } else { // Update
                    targetSpeaker.name = targetName
                    appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, speakers: [...appState.clientSetting.clientSetting.speakers] })
                }
            } else {
                if (targetName.length == 0) { // Noop
                } else {// add
                    appState.clientSetting.clientSetting.speakers.push({
                        id: targetId,
                        name: targetName
                    })
                    appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, speakers: [...appState.clientSetting.clientSetting.speakers] })

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
                        setEditSpeakerTargetName(appState.clientSetting.clientSetting.speakers.find(x => { return x.id == id })?.name || "")
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
    }, [appState.clientSetting.clientSetting.speakers, editSpeakerTargetId, editSpeakerTargetName])

    return editSpeakerIdMappingRow
}