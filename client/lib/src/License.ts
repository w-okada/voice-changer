export type LicenseInfo = {
    name: string,
    url: string
    license: string,
    licenseUrl: string
    description: string
}

export const getLicenceInfo = (): LicenseInfo[] => {
    return [
        {
            name: "VC Helper",
            url: "https://github.com/w-okada/voice-changer",
            license: "MIT",
            licenseUrl: "https://raw.githubusercontent.com/w-okada/voice-changer/master/LICENSE",
            description: ""
        },
        {
            name: "MMVC",
            url: "https://github.com/isletennos/MMVC_Trainer",
            license: "MIT",
            licenseUrl: "https://raw.githubusercontent.com/isletennos/MMVC_Trainer/main/LICENSE",
            description: ""
        },
        {
            name: "so-vits-svc",
            url: "https://github.com/svc-develop-team/so-vits-svc",
            license: "MIT",
            licenseUrl: "https://github.com/svc-develop-team/so-vits-svc/blob/4.0/LICENSE",
            description: ""
        },
        {
            name: "ContentVec",
            url: "https://github.com/auspicious3000/contentvec",
            license: "MIT",
            licenseUrl: "https://raw.githubusercontent.com/auspicious3000/contentvec/main/LICENSE",
            description: ""
        },
    ]
}
