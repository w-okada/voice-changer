import React from "react"
import { useAppRoot } from "./001_provider/001_AppRootProvider"

export const ClientSelector = () => {
    const { setClientType } = useAppRoot()
    return (
        <div>
            <div onClick={() => { setClientType("MMVCv13") }}>MMVCv13</div>
            <div onClick={() => { setClientType("MMVCv15") }}>MMVCv15</div>
            <div onClick={() => { setClientType("so-vits-svc-40") }}>so-vits-svc-40</div>
            <div onClick={() => { setClientType("so-vits-svc-40v2") }}>so-vits-svc-40v2</div>
            <div onClick={() => { setClientType("RVC") }}>RVC</div>
        </div>
    )

}