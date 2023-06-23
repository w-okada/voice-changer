import React, { useEffect, useMemo, useState } from "react";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { useMessageBuilder } from "../../hooks/useMessageBuilder";
import { ModelSlotManagerDialogScreen } from "./904_ModelSlotManagerDialog";

export type EditorScreenProps = {
    screen: ModelSlotManagerDialogScreen
    targetIndex: number
    close: () => void
    backToSlotManager: () => void
}

export const EditorScreen = (props: EditorScreenProps) => {
    const { serverSetting } = useAppState()
    const messageBuilderState = useMessageBuilder()
    const [targetId, setTargetId] = useState<number>(0)
    const [targetName, setTargetName] = useState<string>()

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "header_message", { "ja": "詳細設定: ", "en": "Edit " })
        messageBuilderState.setMessage(__filename, "edit_speaker", { "ja": "話者登録", "en": "Speaker ID" })
        messageBuilderState.setMessage(__filename, "back", { "ja": "戻る", "en": "back" })
    }, [])

    useEffect(() => {
        const targetSlot = serverSetting.serverSetting.modelSlots[props.targetIndex]

        if (!targetSlot) {
            return
        }
        const currentName = !!targetSlot.speakers[targetId] ? targetSlot.speakers[targetId] : ""
        setTargetName(currentName)

    }, [targetId])

    const screen = useMemo(() => {
        if (props.screen != "Editor") {
            return <></>
        }
        const targetSlot = serverSetting.serverSetting.modelSlots[props.targetIndex]




        return (
            <div className="dialog-frame">
                <div className="dialog-title">Model Slot Editor</div>
                <div className="dialog-fixed-size-content">
                    <div className="file-uploader-header">
                        {messageBuilderState.getMessage(__filename, "header_message")} Slot[{props.targetIndex}]
                        <span onClick={() => {
                            props.backToSlotManager()
                        }} className="file-uploader-header-button">&lt;&lt;{messageBuilderState.getMessage(__filename, "back")}</span></div>
                    <div className="edit-model-slot-row">
                        <div className="edit-model-slot-title">
                            {messageBuilderState.getMessage(__filename, "edit_speaker")}
                        </div>
                        <div className="edit-model-slot-speakers">
                            <div className="edit-model-slot-speakers-id-label">
                                ID:
                            </div>
                            <div className="edit-model-slot-speakers-id-select">
                                <select name="" id="" value={targetId} onChange={(e) => { setTargetId(Number(e.target.value)) }}>
                                    {
                                        [...Array(127).keys()].map(x => { return <option key={x} value={x}>{x}</option> })
                                    }
                                </select>
                            </div>
                            <div className="edit-model-slot-speakers-name-label">
                                Name:
                            </div>
                            <div className="edit-model-slot-speakers-name-input">
                                <input id="edit-model-slot-speakers-name-input" value={targetName} onChange={(e) => { setTargetName(e.target.value) }} />
                            </div>
                            <div className="edit-model-slot-speakers-buttons">
                                <div className="edit-model-slot-speakers-button" onClick={async () => {
                                    const inputElem = document.getElementById("edit-model-slot-speakers-name-input") as HTMLInputElement
                                    targetSlot.speakers[targetId] = inputElem.value
                                    await serverSetting.updateModelInfo(props.targetIndex, "speakers", JSON.stringify(targetSlot.speakers))
                                }
                                }>set</div>
                                <div className="edit-model-slot-speakers-button" onClick={async () => {
                                    delete targetSlot.speakers[targetId]
                                    await serverSetting.updateModelInfo(props.targetIndex, "speakers", JSON.stringify(targetSlot.speakers))
                                }
                                }>del</div>
                            </div>
                        </div>
                    </div>
                    <div className="edit-model-slot-row">

                    </div>
                </div>
            </div>
        )

    }, [
        props.screen,
        props.targetIndex,
        targetId,
        targetName
    ])



    return screen;

};
