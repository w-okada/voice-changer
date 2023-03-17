import * as React from "react";
import { useMemo } from "react";
import { useModelSettingArea } from "./102_model_setting";
import { useDeviceSetting } from "./103_device_setting";
import { useConvertSetting } from "./106_convert_setting";
import { useAdvancedSetting } from "./107_advanced_setting";
import { useSpeakerSetting } from "./105_speaker_setting";
import { useServerControl } from "./101_server_control";
import { useQualityControl } from "./104_qulity_control";

export const useMicrophoneOptions = () => {
    const serverControl = useServerControl()
    const modelSetting = useModelSettingArea()
    const deviceSetting = useDeviceSetting()
    const speakerSetting = useSpeakerSetting()
    const convertSetting = useConvertSetting()
    const advancedSetting = useAdvancedSetting()
    const qualityControl = useQualityControl()


    const voiceChangerSetting = useMemo(() => {
        return (
            <>
                {serverControl.serverControl}
                {/* {modelSetting.modelSetting} */}
                {deviceSetting.deviceSetting}
                {qualityControl.qualityControl}
                {speakerSetting.speakerSetting}
                {convertSetting.convertSetting}
                {advancedSetting.advancedSetting}
            </>
        )
    }, [serverControl.serverControl,
    modelSetting.modelSetting,
    deviceSetting.deviceSetting,
    speakerSetting.speakerSetting,
    convertSetting.convertSetting,
    advancedSetting.advancedSetting,
    qualityControl.qualityControl])


    return {
        voiceChangerSetting
    }
}

