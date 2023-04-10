import { useState } from "react"
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
    },
    dialogs: {
        "license": { title: string, auther: string, contact: string, url: string, license: string }[]
    }
}

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
        "advancedSetting": []
    },
    dialogs: {
        "license": [{ title: "", auther: "", contact: "", url: "", license: "MIT" }]

    }
}

export type AppGuiSettingState = {
    appGuiSetting: AppGuiSetting
    guiSettingLoaded: boolean
}

export type AppGuiSettingStateAndMethod = AppGuiSettingState & {
    getAppGuiSetting: (url: string) => Promise<void>
    clearAppGuiSetting: () => void
}

export const userAppGuiSetting = (): AppGuiSettingStateAndMethod => {
    const [guiSettingLoaded, setGuiSettingLoaded] = useState<boolean>(false)
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>(InitialAppGuiDemoSetting)
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
    return {
        appGuiSetting,
        guiSettingLoaded,
        getAppGuiSetting,
        clearAppGuiSetting,

    }
}
