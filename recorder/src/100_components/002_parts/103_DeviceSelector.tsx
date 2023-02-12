import React, { Suspense, useMemo } from "react";
import { useAppSetting } from "../../003_provider/AppSettingProvider";

export const DeviceType = {
    audioinput: "audioinput",
    videoinput: "videoinput",
    audiooutput: "audiooutput",
} as const;
export type DeviceType = typeof DeviceType[keyof typeof DeviceType];

export type DeviceManagerProps = {
    deviceType: DeviceType;
};

export const DeviceSelector = (props: DeviceManagerProps) => {
    const { deviceManagerState } = useAppSetting()

    const targetDevices = useMemo(() => {
        if (props.deviceType === "audioinput") {
            return deviceManagerState.audioInputDevices;
        } else if (props.deviceType === "videoinput") {
            return deviceManagerState.videoInputDevices;
        } else {
            return deviceManagerState.audioOutputDevices;
        }
    }, [deviceManagerState.audioInputDevices, deviceManagerState.videoInputDevices, deviceManagerState.audioOutputDevices]);

    const currentValue = useMemo(() => {
        if (props.deviceType === "audioinput") {
            return deviceManagerState.audioInputDeviceId || "none";
        } else if (props.deviceType === "videoinput") {
            return deviceManagerState.videoInputDeviceId || "none";
        } else {
            return deviceManagerState.audioOutputDeviceId || "none";
        }
    }, [deviceManagerState.audioInputDeviceId, deviceManagerState.videoInputDeviceId, deviceManagerState.audioOutputDeviceId]);

    const setDeviceId = (deviceId: string) => {
        if (props.deviceType === "audioinput") {
            deviceManagerState.setAudioInputDeviceId(deviceId);
        } else if (props.deviceType === "videoinput") {
            deviceManagerState.setVideoInputDeviceId(deviceId);
        } else {
            deviceManagerState.setAudioOutputDeviceId(deviceId);
        }
    };

    const options = useMemo(() => {
        return targetDevices.map((x) => {
            return (
                <option className="device-selector-option" key={x.deviceId} value={x.deviceId}>
                    {x.label}
                </option>
            );
        });
    }, [targetDevices]);

    const selector = useMemo(() => {
        return (
            <select
                defaultValue={currentValue}
                onChange={(e) => {
                    setDeviceId(e.target.value);
                }}
                className="device-selector-select"
            >
                {options}
            </select>
        );
    }, [targetDevices, options, currentValue]);

    const Wrapper = () => {
        if (targetDevices.length === 0) {
            throw new Promise((resolve) => setTimeout(resolve, 1000 * 0.5));
        }
        return selector;
    };
    return (
        <Suspense fallback={<>device loading...</>}>
            <Wrapper></Wrapper>
        </Suspense>
    );
};
