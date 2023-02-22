import { useEffect, useRef, useState } from "react"
import { ApplicationSetting, fetchApplicationSetting, InitialApplicationSetting } from "../001_clients_and_managers/000_ApplicationSettingLoader"

export type ApplicationSettingManagerStateAndMethod = {
    applicationSetting: ApplicationSetting
    setUseMelSpectrogram: (val: boolean) => void
    setCurrentText: (val: string) => void
    setCurrentTextIndex: (val: number) => void
    clearSetting: () => void
}

export const useApplicationSettingManager = (): ApplicationSettingManagerStateAndMethod => {
    const applicationSettingRef = useRef<ApplicationSetting>(InitialApplicationSetting)
    const [applicationSetting, setApplicationSetting] = useState<ApplicationSetting>(applicationSettingRef.current)

    useEffect(() => {
        const url = new URL(window.location.href);
        const params = url.searchParams;
        const settingPath = params.get('setting_path') || null

        const loadApplicationSetting = async () => {

            if (localStorage.applicationSetting) {
                applicationSettingRef.current = JSON.parse(localStorage.applicationSetting) as ApplicationSetting
                console.log("Application setting is loaded from local", applicationSettingRef.current)
                setApplicationSetting({ ...applicationSettingRef.current })
            } else {
                applicationSettingRef.current = await fetchApplicationSetting(settingPath)
                console.log("Application setting is loaded from server", applicationSettingRef.current)
                setApplicationSetting({ ...applicationSettingRef.current })
            }
            setApplicationSetting({ ...applicationSettingRef.current })
        }
        loadApplicationSetting()
    }, [])

    /** (3) Setter */
    /** (3-1) Common */
    const updateApplicationSetting = () => {
        const tmpApplicationSetting = JSON.parse(JSON.stringify(applicationSettingRef.current)) as ApplicationSetting // 大きなデータをリプレースするためのテンポラリ(レコーダでは今のところ不要だが、他のアプリとの処理共通化のため残している。)
        localStorage.applicationSetting = JSON.stringify(tmpApplicationSetting)
        setApplicationSetting({ ...applicationSettingRef.current })
    }
    /** (3-2) Setting */
    const setUseMelSpectrogram = (val: boolean) => {
        applicationSettingRef.current.use_mel_spectrogram = val
        updateApplicationSetting()
    }

    const setCurrentText = (val: string) => {
        applicationSettingRef.current.current_text = val
        updateApplicationSetting()
    }
    const setCurrentTextIndex = (val: number) => {
        applicationSettingRef.current.current_text_index = val
        updateApplicationSetting()
    }

    const clearSetting = () => {
        localStorage.removeItem("applicationSetting")
    }

    return {
        applicationSetting,
        setUseMelSpectrogram,
        setCurrentText,
        setCurrentTextIndex,
        clearSetting,
    }
}

