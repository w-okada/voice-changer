import { AnimationFrameInfo, generateConfig, PSDAnimatorParams, WorkerManager } from "@dannadori/psdanimator"
import { useMemo, useState } from "react"
import { TSUKUYOMI_CANVAS } from "../const"


type PsdAnimationState = {
};

export type PsdAnimationStateAndMethod = PsdAnimationState & {
    loadPsd: (psdFile: string, motionFile: string, speedRate: number) => Promise<void>
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

    const loadPsd = async (psdFile: string, motionFile: string, speedRate: number) => {
        const psd = await (await fetch(psdFile)).arrayBuffer()
        const motion = await (await fetch(motionFile)).json() as AnimationFrameInfo[]
        const canvas = document.getElementById(TSUKUYOMI_CANVAS) as HTMLCanvasElement
        // const c = generateConfig(psdFile, canvas, 640, 480, true)
        const c = generateConfig(psd, canvas, 640, 480, false)
        c.processorURL = "https://cdn.jsdelivr.net/npm/@dannadori/psdanimator@1.0.17/dist/process.js"
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

        const p3: PSDAnimatorParams = {
            type: "SET_WAIT_RATE",
            waitRate: speedRate,
            transfer: []
        }
        await w.execute(p3)
    }


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