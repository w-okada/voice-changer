const StorageTypes = {
    local: "local",
    server: "server",
} as const
export type StorageTypes = typeof StorageTypes[keyof typeof StorageTypes]


export type ApplicationSetting =
    {
        "app_title": string
        "use_mel_spectrogram": boolean
        "storage_type": StorageTypes
        "current_text": string,
        "current_text_index": number,
        "sample_rate": number,
        "text": CorpusTextSetting[]
    }

export type CorpusTextSetting = {
    "title": string,
    "wavPrefix": string,
    "file": string,
    "file_hira": string
}

export const InitialApplicationSetting = require("../../public/assets/setting.json")

export const fetchApplicationSetting = async (settingPath: string | null): Promise<ApplicationSetting> => {
    const url = settingPath || `./assets/setting.json`
    console.log("PATH", settingPath)
    const res = await fetch(url, {
        method: "GET"
    });
    const setting = await res.json() as ApplicationSetting
    return setting;
}
