import React, { useMemo } from "react"
import { fileSelector, Correspondence } from "@dannadori/voice-changer-client-js"

import { useAppState } from "../../001_provider/001_AppStateProvider"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

export const CorrespondenceSelectRow = () => {
    const appState = useAppState()
    const { appGuiSettingState } = useAppRoot()
    const modelSetting = appGuiSettingState.appGuiSetting.front.modelSetting

    const CorrespondenceSelectRow = useMemo(() => {
        if (!modelSetting.MMVCCorrespondense) {
            return <></>
        }
        const correspondenceFileText = appState.clientSetting.clientSetting.correspondences ? JSON.stringify(appState.clientSetting.clientSetting.correspondences.map(x => { return x.dirname })) : ""
        const onCorrespondenceFileLoadClicked = async () => {
            const file = await fileSelector("")

            const correspondenceText = await file.text()
            const cors = correspondenceText.split("\n").map(line => {
                const items = line.split("|")
                if (items.length != 3) {
                    console.warn("Invalid Correspondence Line:", line)
                    return null
                } else {
                    const cor: Correspondence = {
                        sid: Number(items[0]),
                        correspondence: Number(items[1]),
                        dirname: items[2]
                    }
                    return cor
                }
            }).filter(x => { return x != null }) as Correspondence[]
            console.log(cors)
            appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, correspondences: cors })

        }

        const onCorrespondenceFileClearClicked = () => {
            appState.clientSetting.updateClientSetting({ ...appState.clientSetting.clientSetting, correspondences: [] })
        }


        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2">Correspondence</div>
                <div className="body-item-text">
                    <div>{correspondenceFileText}</div>
                </div>
                <div className="body-button-container">
                    <div className="body-button" onClick={onCorrespondenceFileLoadClicked}>select</div>
                    <div className="body-button left-margin-1" onClick={onCorrespondenceFileClearClicked}>clear</div>
                </div>
            </div>
        )
    }, [modelSetting.MMVCCorrespondense, appState.clientSetting.clientSetting.correspondences])

    return CorrespondenceSelectRow
}