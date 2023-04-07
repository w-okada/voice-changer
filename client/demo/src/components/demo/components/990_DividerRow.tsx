import React, { useMemo } from "react"

export type DividerRowProps = {
}

export const DividerRow = (_props: DividerRowProps) => {

    const dividerRow = useMemo(() => {
        return (
            <>
                <div className="body-row divider"></div>
            </>
        )
    }, [])

    return dividerRow
}