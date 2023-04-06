import React, { useMemo } from "react"
import { Title } from "./101_Title"
import { ClearSettingRow } from "./102_ClearSettingRow"

export const TitleArea = () => {
    const titleArea = useMemo(() => {
        return (
            <>
                <Title />
                <ClearSettingRow />
            </>
        )
    }, [])

    return titleArea
}