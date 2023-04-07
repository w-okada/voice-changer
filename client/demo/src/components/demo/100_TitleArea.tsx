import React, { useMemo } from "react"
import { generateComponent } from "./002_ComponentGenerator"
import { useAppRoot } from "../../001_provider/001_AppRootProvider"

export const TitleArea = () => {
    const { appGuiSettingState } = useAppRoot()
    const componentSettings = appGuiSettingState.appGuiSetting.front.title

    const titleArea = useMemo(() => {
        const components = componentSettings.map((x, index) => {
            const c = generateComponent(x.name, x.options)
            return <div key={`${x.name}_${index}`}>{c}</div>
        })
        return (
            <>
                {components}
            </>
        )
    }, [])

    return titleArea
}