import { useIndexedDB } from "@dannadori/voice-changer-client-js"
import React from "react"
import { Title } from "./001-1_Title"
import { useAppRoot } from "./001_provider/001_AppRootProvider"
import { INDEXEDDB_KEY_DEFAULT_MODEL_TYPE } from "./const"

export const ClientSelector = () => {
    const { setClientType } = useAppRoot()
    const { setItem } = useIndexedDB({ clientType: null })
    return (
        <div className="main-body">
            <Title lineNum={1} mainTitle={"Realtime Voice Changer Client"} subTitle={"launcher"} ></Title>
            <div className="body-row split-1-8-1 left-padding-1 ">
                <div></div>
                <div className="body-button-container">
                    <div className="body-button w40 bold" onClick={() => { setClientType("MMVCv13"); setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "MMVCv13") }}>MMVCv13</div>
                    <div className="body-button w40 bold" onClick={() => { setClientType("MMVCv15"); setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "MMVCv15") }}>MMVCv15</div>
                </div>
                <div></div>
            </div>

            <div className="body-row split-1-8-1 left-padding-1 ">
                <div></div>
                <div className="body-button-container">
                    <div className="body-button w40 bold" onClick={() => { setClientType("so-vits-svc-40"); setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "so-vits-svc-40") }}>so-vits-svc-40</div>
                    <div className="body-button w40 bold" onClick={() => { setClientType("so-vits-svc-40v2"); setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "so-vits-svc-40v2") }}>so-vits-svc-40v2</div>
                </div>
                <div></div>
            </div>

            <div className="body-row split-1-8-1 left-padding-1 ">
                <div></div>
                <div className="body-button-container">
                    <div className="body-button w40 bold" onClick={() => { setClientType("RVC"); setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "RVC") }}>RVC</div>
                    <div className="body-button w40 bold" onClick={() => { setClientType("DDSP-SVC"); setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "DDSP-SVC") }}>DDSP-SVC(N/A)</div>
                </div>
                <div></div>
            </div>
        </div>
    )

}