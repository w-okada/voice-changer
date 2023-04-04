import { useState } from "react"
import { AppGuiDemoClearSetting } from "../components/demo/102_ClearSettingRow"
import { AppGuiDemoDialogLicenseSetting } from "../components/demo/901_LicenseDialog"
import { ClientType } from "@dannadori/voice-changer-client-js"

export type AppGuiSetting = AppGuiDemoSetting

export type AppGuiDemoSetting = {
    type: "demo",
    id: ClientType,
    front: {
        "title": {
            "mainTitle": string,
            "subTitle": string,
            "lineNum": number
        },
        "serverControl": {
        },
        "modelSetting": {
            "ONNXEnable": boolean,
            "pyTorchEnable": boolean,
            "MMVCCorrespondense": boolean,
            "pyTorchClusterEnable": boolean,
        },
        "deviceSetting": {},
        "qualityControl": {
            "F0DetectorEnable": boolean
        },
        "speakerSetting": AppGuiDemoComponents[],
        "converterSetting": AppGuiDemoComponents[],
        "advancedSetting": AppGuiDemoComponents[]
    },
    dialogs: {
        "license": { title: string, auther: string, contact: string, url: string, license: string }[]
    }
}

export type AppGuiDemoComponents = AppGuiDemoClearSetting
export type AppGuiDemoDialogComponents = AppGuiDemoDialogLicenseSetting



const InitialAppGuiDemoSetting: AppGuiDemoSetting = {
    type: "demo",
    id: ClientType.MMVCv13,
    front: {
        "title": {
            "mainTitle": "",
            "subTitle": "",
            "lineNum": 1
        },
        "serverControl": {

        },
        "modelSetting": {
            "ONNXEnable": false,
            "pyTorchEnable": false,
            "MMVCCorrespondense": false,
            "pyTorchClusterEnable": false,
        },
        "deviceSetting": {},
        "qualityControl": {
            "F0DetectorEnable": false
        },
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
}

export type AppGuiSettingStateAndMethod = AppGuiSettingState & {
    getAppSetting: (url: string) => Promise<void>
}

export const userAppGuiSetting = (): AppGuiSettingStateAndMethod => {
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>(InitialAppGuiDemoSetting)
    const getAppSetting = async (url: string) => {
        const res = await fetch(`${url}`, {
            method: "GET",
        })
        const appSetting = await res.json() as AppGuiSetting
        setAppGuiSetting(appSetting)
    }
    return {
        appGuiSetting,
        getAppSetting,
    }
}
