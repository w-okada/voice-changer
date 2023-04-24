import { ClientType, useIndexedDB } from "@dannadori/voice-changer-client-js"
import React, { useMemo } from "react"
import { Title } from "./001-1_Title"
import { useAppRoot } from "./001_provider/001_AppRootProvider"
import { INDEXEDDB_KEY_DEFAULT_MODEL_TYPE } from "./const"

export const ClientSelector = () => {
    const { setClientType } = useAppRoot()
    const { setItem } = useIndexedDB({ clientType: null })
    const onClientTypeClicked = (clientType: ClientType) => {
        setClientType(clientType);
        setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, clientType)
    }



    const selectableClientTypes = useMemo(() => {
        const ua = window.navigator.userAgent.toLowerCase();
        if (ua.indexOf("mac os x") !== -1) {
            return ["MMVCv13", "MMVCv15", "so-vits-svc-40", "RVC"] as ClientType[]
        } else {
            return ["MMVCv13", "MMVCv15", "so-vits-svc-40", "so-vits-svc-40v2", "RVC", "DDSP-SVC", "RVC_CLASSIC_GUI"] as ClientType[]
        }
    }, [])

    const selectableClientTypesRowItems = useMemo(() => {
        return selectableClientTypes.flatMap((_, i, a) => { return i % 2 ? [] : [a.slice(i, i + 2)] })
    }, [])

    const selectableClientTypesRow = useMemo(() => {
        return selectableClientTypesRowItems.map((x, index) => {
            return (
                <div key={index} className="body-row split-1-8-1 left-padding-1 ">
                    <div></div>
                    <div className="body-button-container">

                        {
                            x.map(y => {
                                return <div key={y} className="body-button w40 bold" onClick={() => { onClientTypeClicked(y) }}>{y}</div>
                            })
                        }
                    </div>
                    <div></div>
                </div>
            )
        })
    }, [])

    return (

        <div className="main-body">
            <Title lineNum={1} mainTitle={"Realtime Voice Changer Client"} subTitle={"launcher"} ></Title>
            {selectableClientTypesRow}
        </div>
    )

}