import { useEffect, useState } from "react"

export type AppSettings = {
    charaName: string,
    psdFile: string,
    motionFile: string,
    motionSpeedRate: number
}

const InitialAppSettings: AppSettings = {
    charaName: "",
    psdFile: "",
    motionFile: "",
    motionSpeedRate: 1
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