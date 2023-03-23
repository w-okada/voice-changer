import { AnimationFrameInfo, generateConfig, PSDAnimatorParams, WorkerManager } from "@dannadori/psdanimator"
import { useMemo, useState } from "react"
import { TSUKUYOMI_CANVAS } from "../const"


type PsdAnimationState = {
};

export type PsdAnimationStateAndMethod = PsdAnimationState & {
    loadPsd: (psdFile: string, motionFile: string) => Promise<void>
    start: () => Promise<void>
    switchNormalMotion: () => Promise<void>
    switchTalkingMotion: () => Promise<void>
    psdAnimationInitialized: boolean
}

export const usePsdAnimation = (): PsdAnimationStateAndMethod => {
    const w = useMemo(() => {
        return new WorkerManager()
    }, [])
    const [psdAnimationInitialized, setPsdAnimationInitialized] = useState<boolean>(false)

    const loadPsd = async (psdFile: string, motionFile: string) => {
        const psd = await (await fetch(psdFile)).arrayBuffer()
        const motion = await (await fetch(motionFile)).json() as AnimationFrameInfo[]
        const canvas = document.getElementById(TSUKUYOMI_CANVAS) as HTMLCanvasElement
        // const c = generateConfig(psdFile, canvas, 640, 480, true)
        const c = generateConfig(psd, canvas, 640, 480, false)
        c.transfer = [c.canvas]
        await w.init(c)
        console.log("[psd animator] Initialized")

        const p1: PSDAnimatorParams = {
            type: "SET_MOTION",
            motion: motion,
            transfer: []
        }
        await w.execute(p1)
        console.log("[psd animator] Set motion")
        setPsdAnimationInitialized(true)
    }


    // "ROOT_!眉　_*通常眉",
    // "ROOT_!眉　_*通常眉：少し下げ",
    // "ROOT_!眉　_*喜び・驚き",
    // "ROOT_!目_*通常目",
    // "ROOT_!目_*目を少し細める ★デフォルト",
    // "ROOT_!目_*目を細める",
    // "ROOT_!目_*目を閉じる",
    // "ROOT_!目_*目を閉じて微笑む",
    // "ROOT_!頬　_*通常",
    // "ROOT_!口 _*口を閉じて微笑む",
    // "ROOT_!口 _*小さく笑う",
    // "ROOT_!口 _*大きく笑う",
    // "ROOT_!髪（おさげ）_!髪（おさげ）　※選択肢なし",
    // "ROOT_!左腕_*左手を下げる",
    // "ROOT_!左腕_*左手でご案内",
    // "ROOT_!左腕_*左手を胸の前に：グー",
    // "ROOT_!左腕_*左手を胸の前に：パー",
    // "ROOT_!左腕_*左手を口に当てる",
    // "ROOT_!右腕_*右手を下げる",
    // "ROOT_!右腕_*右手でご案内",
    // "ROOT_!右腕_*右手を胸の前に：グー",
    // "ROOT_!右腕_*右手を胸の前に：パー",
    // "ROOT_!右腕_*右手を口に当てる",
    // "ROOT_!体_!体　※選択肢なし"

    const start = async () => {
        const p3: PSDAnimatorParams = {
            type: "START",
            transfer: []
        }
        await w.execute(p3)
    }
    const switchNormalMotion = async () => {
        const p2: PSDAnimatorParams = {
            type: "SWITCH_MOTION_MODE",
            motionMode: "normal",
            transfer: []
        }
        await w.execute(p2)
    }
    const switchTalkingMotion = async () => {
        const p2: PSDAnimatorParams = {
            type: "SWITCH_MOTION_MODE",
            motionMode: "talking",
            transfer: []
        }
        await w.execute(p2)
    }
    return {
        loadPsd,
        switchNormalMotion,
        switchTalkingMotion,
        start,
        psdAnimationInitialized
    }
}