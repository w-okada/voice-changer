import React, { useMemo, useState } from "react"
import { useAppState } from "../../../001_provider/001_AppStateProvider"
import { useGuiState } from "../001_GuiStateProvider"

export type SampleModelSelectRowProps = {}
export const SampleModelSelectRow = (_props: SampleModelSelectRowProps) => {
    const appState = useAppState()
    const guiState = useGuiState()
    const [lang, setLang] = useState<string>("All")
    const sampleModelSelectRow = useMemo(() => {
        const slot = guiState.modelSlotNum
        const fileUploadSetting = appState.serverSetting.fileUploadSettings[slot]
        if (!fileUploadSetting) {
            return <></>
        }
        if (fileUploadSetting.isSampleMode == false) {
            return <></>
        }

        const langs = appState.serverSetting.serverSetting.sampleModels.reduce((prev, cur) => {
            if (prev.includes(cur.lang) == false) {
                prev.push(cur.lang)
            }
            return prev
        }, ["All"] as string[])

        const langOptions = (
            langs.map(x => {
                return <option key={x} value={x}>{x}</option>
            })
        )

        const options = (
            appState.serverSetting.serverSetting.sampleModels.filter(x => { return lang == "All" ? true : x.lang == lang }).map(x => {
                return <option key={x.id} value={x.id}>{x.name}</option>
            })
        )


        const selectedSample = appState.serverSetting.serverSetting.sampleModels.find(x => { return x.id == fileUploadSetting.sampleId })
        const creditText = selectedSample ? `credit:${selectedSample.credit}` : ""
        const termOfUseLink = selectedSample ? <a href={selectedSample.termsOfUseUrl} target="_blank" rel="noopener noreferrer" className="body-item-text-small">[terms of use]</a> : <></>

        const onModelSelected = (val: string) => {
            appState.serverSetting.setFileUploadSetting(slot, {
                ...appState.serverSetting.fileUploadSettings[slot], sampleId: val
            })
        }

        return (
            <div className="body-row split-3-3-4 left-padding-1 guided">
                <div className="body-item-title left-padding-2 ">Select Model</div>
                <div>
                    <select value={lang} onChange={(e) => { setLang(e.target.value) }}>
                        {langOptions}
                    </select>
                    <select value={fileUploadSetting.sampleId || ""} onChange={(e) => { onModelSelected(e.target.value) }}>
                        <option disabled value={""}> -- select model -- </option>
                        {options}
                    </select>
                </div>
                <div className="body-item-text">
                    {creditText}{termOfUseLink}
                </div>
            </div>
        )

    }, [appState.serverSetting.fileUploadSettings, guiState.modelSlotNum, lang])

    return sampleModelSelectRow
}