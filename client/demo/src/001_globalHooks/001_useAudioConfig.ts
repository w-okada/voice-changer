import { useEffect, useState } from "react"

export type AudioConfigState = {
    audioContext: AudioContext | null
}
export const useAudioConfig = (): AudioConfigState => {
    const [audioContext, setAudioContext] = useState<AudioContext | null>(null)

    useEffect(() => {
        const createAudioContext = () => {

            const url = new URL(window.location.href);
            const params = url.searchParams;
            const sampleRate = params.get('sample_rate') || null
            let ctx: AudioContext
            if(sampleRate){
                if(sampleRate == "default"){
                    console.log(`Sample rate: default`)
                    ctx = new AudioContext()
                }else{
                    console.log(`Sample rate: ${sampleRate}`)
                    ctx = new AudioContext({ sampleRate: Number(sampleRate)})
                }
            }else{
                console.log(`Sample rate: default(48000)`)
                ctx = new AudioContext({ sampleRate: 48000})
            }

            console.log(ctx)
            setAudioContext(ctx)

            document.removeEventListener('touchstart', createAudioContext);
            document.removeEventListener('mousedown', createAudioContext);
        }
        document.addEventListener('touchstart', createAudioContext, false);
        document.addEventListener('mousedown', createAudioContext, false);
    }, [])

    const ret: AudioConfigState = {
        audioContext
    }

    return ret

}