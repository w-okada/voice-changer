import { useEffect, useState } from "react"

export type AudioConfigState = {
    audioContext: AudioContext | null
}

export const useAudioConfig = (): AudioConfigState => {
    const [audioContext, setAudioContext] = useState<AudioContext | null>(null)
    useEffect(() => {
        const createAudioContext = () => {
            const ctx = new AudioContext()
            document.removeEventListener('touchstart', createAudioContext);
            document.removeEventListener('mousedown', createAudioContext);
            setAudioContext(ctx)
        }
        document.addEventListener('touchstart', createAudioContext, false);
        document.addEventListener('mousedown', createAudioContext, false);
    }, [])

    const ret: AudioConfigState = {
        audioContext
    }

    return ret

}