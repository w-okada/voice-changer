import { useEffect, useState } from "react"
import { ClientType } from "@dannadori/voice-changer-client-js"

export type AppGuiSetting = AppGuiDemoSetting

export type AppGuiDemoSetting = {
    type: "demo",
    id: ClientType,
    front: {
        "modelSlotControl": GuiComponentSetting[],
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
        "modelSlotControl": []
    }
}

export type AppGuiSettingState = {
    appGuiSetting: AppGuiSetting
    guiSettingLoaded: boolean
    version: string
    edition: string
}

export type AppGuiSettingStateAndMethod = AppGuiSettingState & {
    getAppGuiSetting: (url: string) => Promise<void>
    clearAppGuiSetting: () => void
}

export const useAppGuiSetting = (): AppGuiSettingStateAndMethod => {
    const [guiSettingLoaded, setGuiSettingLoaded] = useState<boolean>(false)
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>(InitialAppGuiDemoSetting)
    const [version, setVersion] = useState<string>("")
    const [edition, setEdition] = useState<string>("")
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

    useEffect(() => {
        const getVersionInfo = async () => {
            const res = await fetch(`/assets/gui_settings/edition.txt`, {
                method: "GET",
            })
            const edition = await res.text()
            setEdition(edition)
        }
        getVersionInfo()
    }, [])

    return {
        appGuiSetting,
        guiSettingLoaded,
        version,
        edition,
        getAppGuiSetting,
        clearAppGuiSetting,
    }
}

