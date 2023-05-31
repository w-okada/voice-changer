
const messages: {
    [id: string]: {
        [lang: string]: string
    }
} = {
    "notice_1": {
        "en": "DirectML version is an experimental version. There are the known issues as follows.",
        "ja": "directML版は実験的バージョンです。以下の既知の問題があります。",
    },
    "notice_2": {
        "en": "(1) When some settings are changed, conversion process becomes slow even when using GPU. If this occurs, reset the GPU value to -1 and then back to 0.",
        "ja": "(1) 一部の設定変更を行うとgpuを使用していても変換処理が遅くなることが発生します。もしこの現象が発生したらGPUの値を-1にしてから再度0に戻してください。",
    },
    "donate_1": {
        "en": "This software is supported by donations. Thank you for your support!",
        "ja": "開発者にコーヒーをご馳走してあげよう。この黄色いアイコンから。",
    },
    "click_to_start_1": {
        "en": "Click to start",
        "ja": "スタートボタンを押してください。",
    }

}

export const getMessage = (id: string) => {
    let lang = window.navigator.language
    if (lang != "ja") {
        lang = "en"
    }

    if (!messages[id]) {
        return "undefined message."
    }

    return messages[id][lang]
}