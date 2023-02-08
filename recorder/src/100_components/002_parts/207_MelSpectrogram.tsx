import React, { useEffect, useMemo } from "react"
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { useAppState } from "../../003_provider/AppStateProvider";
import { drawMel } from "../../const";

const MelTypes = {
    mic: "mic",
    vf: "vf"
} as const
type MelTypes = typeof MelTypes[keyof typeof MelTypes]

export const MelSpectrogram = () => {
    const { applicationSetting } = useAppSetting()
    const { audioControllerState } = useAppState();

    useEffect(() => {
        const drawMel = (type: MelTypes, data: string) => {
            let canvas = document.getElementById("mel") as HTMLCanvasElement

            if (type === "mic") {
                canvas = document.getElementById("mel_mic") as HTMLCanvasElement
            } else {
                canvas = document.getElementById("mel_vf") as HTMLCanvasElement
            }
            const ctx2d = canvas.getContext("2d")!
            if (!data) {
                ctx2d.clearRect(0, 0, canvas.width, canvas.height)
                return
            }


            const img = new Image()
            img.src = data
            img.onload = () => {
                console.log("ONLOAD")
                ctx2d.drawImage(img, 0, 0, canvas.width, canvas.height)
            }
        }

        if (audioControllerState.tempUserData.micSpec.length == 0) {
            return
        }

        if (applicationSetting.applicationSetting.use_mel_spectrogram) {
            drawMel("mic", audioControllerState.tempUserData.micSpec)
            drawMel("vf", audioControllerState.tempUserData.vfSpec)
        }

    }, [audioControllerState.tempUserData.micSpec, audioControllerState.tempUserData.vfSpec, applicationSetting.applicationSetting?.use_mel_spectrogram])

    const mel = useMemo(() => {
        if (!applicationSetting.applicationSetting.use_mel_spectrogram) {
            return <></>
        }
        if (!audioControllerState.tempUserData.vfWavBlob ||
            !audioControllerState.tempUserData.micWavBlob
        ) {
            return <></>
        }

        if (audioControllerState.tempUserData.micSpec.length == 0) {
            const generateMelSpec = () => {
                if (
                    !audioControllerState.tempUserData.vfWavSamples ||
                    !audioControllerState.tempUserData.micWavSamples
                ) {
                    return
                }


                const micSpec = drawMel(audioControllerState.tempUserData.micWavSamples, applicationSetting.applicationSetting.sample_rate)
                const vfSpec = drawMel(audioControllerState.tempUserData.vfWavSamples, applicationSetting.applicationSetting.sample_rate)
                audioControllerState.setTempWavBlob(
                    audioControllerState.tempUserData.micWavBlob,
                    audioControllerState.tempUserData.vfWavBlob,
                    audioControllerState.tempUserData.micWavSamples,
                    audioControllerState.tempUserData.vfWavSamples,
                    micSpec,
                    vfSpec,
                    audioControllerState.tempUserData.region!)
                audioControllerState.setUnsavedRecord(true);
            }
            return (
                <div onClick={generateMelSpec} className="button">Generate MelSpec</div>
            )
        }

        return (
            <>
                <div className="mel-spectrogram-div">
                    <canvas id="mel_mic" className="mel-spectrogram-canvas"></canvas>
                </div>
                <div className="mel-spectrogram-div">
                    <canvas id="mel_vf" className="mel-spectrogram-canvas"></canvas>
                </div>
            </>
        )

    }, [audioControllerState.tempUserData.micSpec, audioControllerState.tempUserData.vfSpec, audioControllerState.tempUserData.vfWavBlob, applicationSetting.applicationSetting.use_mel_spectrogram])
    return (
        <>
            {mel}
        </>
    )
}