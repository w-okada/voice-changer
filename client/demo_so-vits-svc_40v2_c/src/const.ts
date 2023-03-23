import { ClientType } from "@dannadori/voice-changer-client-js"

export const CLIENT_TYPE = ClientType.so_vits_svc_40v2_c

export const AUDIO_ELEMENT_FOR_PLAY_RESULT = "audio-result"
export const AUDIO_ELEMENT_FOR_TEST_ORIGINAL = "audio-test-original"
export const AUDIO_ELEMENT_FOR_TEST_CONVERTED = "audio-test-converted"
export const AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK = "audio-test-converted-echoback"


export const INDEXEDDB_KEY_AUDIO_OUTPUT = "INDEXEDDB_KEY_AUDIO_OUTPUT"


// State Control Checkbox
export const OpenServerControlCheckbox = "open-server-control-checkbox"
export const OpenModelSettingCheckbox = "open-model-setting-checkbox"
export const OpenDeviceSettingCheckbox = "open-device-setting-checkbox"
export const OpenQualityControlCheckbox = "open-quality-control-checkbox"
export const OpenSpeakerSettingCheckbox = "open-speaker-setting-checkbox"
export const OpenConverterSettingCheckbox = "open-converter-setting-checkbox"
export const OpenAdvancedSettingCheckbox = "open-advanced-setting-checkbox"

export const isDesktopApp = () => {
    if (navigator.userAgent.indexOf('Electron') >= 0) {
        return true;
    } else {
        return false;
    }
};


// tsukuyomi
export const TSUKUYOMI_CANVAS = "tsukuyomi-canvas"