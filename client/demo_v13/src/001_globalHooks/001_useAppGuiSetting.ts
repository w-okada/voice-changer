import { useState } from "react"
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
            "showPyTorchDefault": boolean
        },
        "deviceSetting": {},
        "qualityControl": {
            "F0DetectorEnable": boolean
        },
        "speakerSetting": {
            "srcIdEnable": boolean
            "editSpeakerIdMappingEnable": boolean
            "f0FactorEnable": boolean
            "tuningEnable": boolean
            "clusterInferRationEnable": boolean
            "noiseScaleEnable": boolean
            "silentThresholdEnable": boolean
        },
        "converterSetting": {
            "extraDataLengthEnable": boolean
        },
        "advancedSetting": {
            "serverURLEnable": boolean,
            "protocolEnable": boolean,
            "sampleRateEnable": boolean,
            "sendingSampleRateEnable": boolean,
            "crossFadeOverlapSizeEnable": boolean,
            "crossFadeOffsetRateEnable": boolean,
            "crossFadeEndRateEnable": boolean,
            "downSamplingModeEnable": boolean,
            "trancateNumTresholdEnable": boolean,
        }
    },
    dialogs: {
        "license": { title: string, auther: string, contact: string, url: string, license: string }[]
    }
}



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
            "showPyTorchDefault": false
        },
        "deviceSetting": {},
        "qualityControl": {
            "F0DetectorEnable": false
        },
        "speakerSetting": {
            "srcIdEnable": false,
            "editSpeakerIdMappingEnable": false,
            "f0FactorEnable": false,
            "tuningEnable": false,
            "clusterInferRationEnable": false,
            "noiseScaleEnable": false,
            "silentThresholdEnable": false

        },
        "converterSetting": {
            "extraDataLengthEnable": false
        },
        "advancedSetting": {
            "serverURLEnable": false,
            "protocolEnable": false,
            "sampleRateEnable": false,
            "sendingSampleRateEnable": false,
            "crossFadeOverlapSizeEnable": false,
            "crossFadeOffsetRateEnable": false,
            "crossFadeEndRateEnable": false,
            "downSamplingModeEnable": false,
            "trancateNumTresholdEnable": false,
        }
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
    getAppSetting: (url: string) => Promise<void>
}

export const userAppGuiSetting = (): AppGuiSettingStateAndMethod => {
    const [guiSettingLoaded, setGuiSettingLoaded] = useState<boolean>(false)
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>(InitialAppGuiDemoSetting)
    const getAppSetting = async (url: string) => {
        const res = await fetch(`${url}`, {
            method: "GET",
        })
        const appSetting = await res.json() as AppGuiSetting
        setAppGuiSetting(appSetting)
        setGuiSettingLoaded(true)
    }
    return {
        appGuiSetting,
        guiSettingLoaded,
        getAppSetting,
    }
}
