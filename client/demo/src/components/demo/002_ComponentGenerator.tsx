import React from "react"
import { ModelSlotArea, ModelSlotAreaProps } from "./components2/100_ModelSlotArea"
import { CharacterArea, CharacterAreaProps } from "./components2/101_CharacterArea"
import { ConfigArea, ConfigAreaProps } from "./components2/102_ConfigArea"
import { HeaderArea, HeaderAreaProps } from "./components2/001_HeaderArea"

export const catalog: { [key: string]: (props: any) => JSX.Element } = {}

export const addToCatalog = (key: string, generator: (props: any) => JSX.Element) => {
    catalog[key] = generator
}

export const generateComponent = (key: string, props: any) => {
    if (!catalog[key]) {
        console.error("not found component generator.", key)
        return <></>
    }
    return catalog[key](props)
}

const initialize = () => {
    addToCatalog("headerArea", (props: HeaderAreaProps) => { return <HeaderArea {...props} /> })
    addToCatalog("modelSlotArea", (props: ModelSlotAreaProps) => { return <ModelSlotArea {...props} /> })
    addToCatalog("characterArea", (props: CharacterAreaProps) => { return <CharacterArea {...props} /> })
    addToCatalog("configArea", (props: ConfigAreaProps) => { return <ConfigArea {...props} /> })

}

initialize()