import { useEffect, useState } from "react"

export type AppSettings = {
    charaName: string,
    psdFile: string,
    motionFile: string,
}

const InitialAppSettings: AppSettings = {
    charaName: "",
    psdFile: "",
    motionFile: "",
}

export type AppSettingStates = {
    appSettings: AppSettings

}

export const useAppSettings = (): AppSettingStates => {
    const [appSettings, setAppSettings] = useState<AppSettings>(InitialAppSettings)
    useEffect(() => {
        const loadAppSettings = async () => {
            const ret = await (await fetch("/assets/settings/settings.json")).json() as AppSettings
            setAppSettings(ret)
        }
        loadAppSettings()
    }, [])


    return {
        appSettings
    }

}