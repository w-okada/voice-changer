import { useState, useEffect } from "react";
import { WorkletSetting } from "../const";
import { VoiceChangerClient } from "../VoiceChangerClient";

export type UseWorkletSettingProps = {
    voiceChangerClient: VoiceChangerClient | null;
    workletSetting: WorkletSetting;
};

export type WorkletSettingState = {
    // setting: WorkletSetting;
    // setSetting: (setting: WorkletSetting) => void;
};

export const useWorkletSetting = (props: UseWorkletSettingProps): WorkletSettingState => {
    const [setting, _setSetting] = useState<WorkletSetting>(props.workletSetting);

    useEffect(() => {
        if (!props.voiceChangerClient) return;
        props.voiceChangerClient.configureWorklet(setting);
    }, [props.voiceChangerClient, props.workletSetting]);

    // // 設定 _setSettingがトリガでuseEffectが呼ばれて、workletに設定が飛ぶ
    // const setSetting = useMemo(() => {
    //     return (setting: WorkletSetting) => {
    //         if (!props.voiceChangerClient) return
    //         _setSetting(setting)
    //     }
    // }, [props.voiceChangerClient])

    return {
        // setting,
        // setSetting,
    };
};
