
export const AUDIO_ELEMENT_FOR_PLAY_RESULT = "audio-result"  // 変換後の出力用プレイヤー
export const AUDIO_ELEMENT_FOR_PLAY_MONITOR = "audio-monitor"  //  変換後のモニター用プレイヤー
export const AUDIO_ELEMENT_FOR_TEST_ORIGINAL = "audio-test-original"  // ??? 使ってないかも。
export const AUDIO_ELEMENT_FOR_TEST_CONVERTED = "audio-test-converted" // ファイルインプットのコントロール
export const AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK = "audio-test-converted-echoback"  // ファイルインプットのエコーバック

export const AUDIO_ELEMENT_FOR_SAMPLING_INPUT = "body-wav-container-wav-input"
export const AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT = "body-wav-container-wav-output"

export const INDEXEDDB_KEY_AUDIO_OUTPUT = "INDEXEDDB_KEY_AUDIO_OUTPUT"
export const INDEXEDDB_KEY_AUDIO_MONITR = "INDEXEDDB_KEY_AUDIO_MONITOR"
export const INDEXEDDB_KEY_DEFAULT_MODEL_TYPE = "INDEXEDDB_KEY_DEFALT_MODEL_TYPE"

export const MODEL_ICON_BLANK_URL = "/assets/icons/blank.png"

export const isDesktopApp = () => {
    if (navigator.userAgent.indexOf('Electron') >= 0) {
        return true;
    } else {
        return false;
    }
};




