import { ClientType } from "@dannadori/voice-changer-client-js"

export const CLIENT_TYPE = ClientType.MMVCv13

export const AUDIO_ELEMENT_FOR_PLAY_RESULT = "audio-result"
export const AUDIO_ELEMENT_FOR_TEST_ORIGINAL = "audio-test-original"
export const AUDIO_ELEMENT_FOR_TEST_CONVERTED = "audio-test-converted"
export const AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK = "audio-test-converted-echoback"

export const AUDIO_ELEMENT_FOR_SAMPLING_INPUT = "body-wav-container-wav-input"
export const AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT = "body-wav-container-wav-output"

export const INDEXEDDB_KEY_AUDIO_OUTPUT = "INDEXEDDB_KEY_AUDIO_OUTPUT"


export const isDesktopApp = () => {
    if (navigator.userAgent.indexOf('Electron') >= 0) {
        return true;
    } else {
        return false;
    }
};


