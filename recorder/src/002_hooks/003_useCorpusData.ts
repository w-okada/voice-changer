import { useEffect, useState } from "react"
import { fetchTextResource } from "../001_clients_and_managers/002_ResourceLoader"
import { useAppSetting } from "../003_provider/AppSettingProvider"

export type CorpusTextData = {
    "title": string,
    "wavPrefix": string,
    "file": string,
    "file_hira": string,
    "text": string[]
    "text_hira": string[]
}

export type CorpusDataState = {
    corpusTextData: { [title: string]: CorpusTextData }
    corpusLoaded: boolean
}
export type CorpusDataStateAndMethod = CorpusDataState & {}

export const useCorpusData = (): CorpusDataStateAndMethod => {
    const { applicationSetting } = useAppSetting()
    const textSettings = applicationSetting.applicationSetting.text
    const [corpusTextData, setCorpusTextData] = useState<{ [title: string]: CorpusTextData }>({})
    const [corpusLoaded, setCorpusLoaded] = useState<boolean>(false)


    useEffect(() => {
        if (!textSettings) {
            return
        }
        const loadCorpusText = async () => {
            const newCorpusTextData: { [title: string]: CorpusTextData } = {}
            for (const x of textSettings) {
                const text = await fetchTextResource(x.file)
                const textHira = await fetchTextResource(x.file_hira)
                const splitText = text.split("\n").filter(x => { return x.length > 0 })
                const splitTextHira = textHira.split("\n").filter(x => { return x.length > 0 })

                const data: CorpusTextData = {
                    title: x.title,
                    wavPrefix: x.wavPrefix,
                    file: x.file,
                    file_hira: x.file_hira,
                    text: splitText,
                    text_hira: splitTextHira,
                }
                newCorpusTextData[data.title] = data
            }
            setCorpusTextData(newCorpusTextData)
            setCorpusLoaded(true)
        }
        loadCorpusText()
    }, [textSettings])


    return {
        corpusTextData,
        corpusLoaded,
    }
}
