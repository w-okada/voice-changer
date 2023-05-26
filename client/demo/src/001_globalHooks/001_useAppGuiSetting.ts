import { useEffect, useState } from "react"
import { ClientType } from "@dannadori/voice-changer-client-js"

export type AppGuiSetting = AppGuiDemoSetting

export type AppGuiDemoSetting = {
    type: "demo",
    id: ClientType,
    front: {
        "title": GuiComponentSetting[],
        "serverControl": GuiComponentSetting[],
        "modelSetting": GuiComponentSetting[],
        "deviceSetting": GuiComponentSetting[],
        "qualityControl": GuiComponentSetting[],
        "speakerSetting": GuiComponentSetting[],
        "converterSetting": GuiComponentSetting[],
        "advancedSetting": GuiComponentSetting[],
        "lab": GuiComponentSetting[],
    },
    dialogs: {
        "license": { title: string, auther: string, contact: string, url: string, license: string }[]
    }
}

// export type AppGuiDemoSetting2 = {
//     type: "demo",
//     id: ClientType,
//     front: GuiSectionSetting[],
//     dialogs: {
//         "license": { title: string, auther: string, contact: string, url: string, license: string }[]
//     }
// }


// export type GuiSectionSetting = {
//     "title": string,
//     "components": GuiComponentSetting[]
// }

export type GuiComponentSetting = {
    "name": string,
    "options": any
}

const InitialAppGuiDemoSetting: AppGuiDemoSetting = {
    type: "demo",
    id: ClientType.MMVCv13,
    front: {
        "title": [],
        "serverControl": [],
        "modelSetting": [],
        "deviceSetting": [],
        "qualityControl": [],
        "speakerSetting": [],
        "converterSetting": [],
        "advancedSetting": [],
        "lab": []
    },
    dialogs: {
        "license": [{ title: "", auther: "", contact: "", url: "", license: "MIT" }]

    }
}

export type AppGuiSettingState = {
    appGuiSetting: AppGuiSetting
    guiSettingLoaded: boolean
    version: string
}

export type AppGuiSettingStateAndMethod = AppGuiSettingState & {
    getAppGuiSetting: (url: string) => Promise<void>
    clearAppGuiSetting: () => void
}

export const useAppGuiSetting = (): AppGuiSettingStateAndMethod => {
    const [guiSettingLoaded, setGuiSettingLoaded] = useState<boolean>(false)
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>(InitialAppGuiDemoSetting)
    const [version, setVersion] = useState<string>("")
    const getAppGuiSetting = async (url: string) => {
        const res = await fetch(`${url}`, {
            method: "GET",
        })
        const appSetting = await res.json() as AppGuiSetting
        setAppGuiSetting(appSetting)
        setGuiSettingLoaded(true)
    }
    const clearAppGuiSetting = () => {
        setAppGuiSetting(InitialAppGuiDemoSetting)
        setGuiSettingLoaded(false)
    }

    useEffect(() => {
        const getVersionInfo = async () => {
            const res = await fetch(`/assets/gui_settings/version.txt`, {
                method: "GET",
            })
            const version = await res.text()
            setVersion(version)
        }
        getVersionInfo()
    }, [])

    return {
        appGuiSetting,
        guiSettingLoaded,
        version,
        getAppGuiSetting,
        clearAppGuiSetting,
    }
}

