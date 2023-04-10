import React, { useMemo } from "react";
import { isDesktopApp } from "./const";


export type TitleProps = {
    lineNum: number
    mainTitle: string
    subTitle: string
}

export const Title = (props: TitleProps) => {


    const githubLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://github.com/w-okada/voice-changer") }}>
                    <img src="./assets/icons/github.svg" />
                    <div className="tooltip-text">github</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                    <img src="./assets/icons/github.svg" />
                    <div className="tooltip-text">github</div>
                </a>
            )
    }, [])


    const manualLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://zenn.dev/wok/books/0003_vc-helper-v_1_5") }}>
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text">manual</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://zenn.dev/wok/books/0003_vc-helper-v_1_5" target="_blank" rel="noopener noreferrer">
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text">manual</div>
                </a>
            )
    }, [])


    const toolLink = useMemo(() => {
        return isDesktopApp() ?
            (
                <div className="link tooltip">
                    <img src="./assets/icons/tool.svg" />
                    <div className="tooltip-text tooltip-text-100px">
                        <p onClick={() => {
                            // @ts-ignore
                            window.electronAPI.openBrowser("https://w-okada.github.io/screen-recorder-ts/")
                        }}>
                            screen capture
                        </p>
                    </div>
                </div>
            )
            :
            (
                <div className="link tooltip">
                    <img src="./assets/icons/tool.svg" />
                    <div className="tooltip-text tooltip-text-100px">
                        <p onClick={() => {
                            window.open("https://w-okada.github.io/screen-recorder-ts/", '_blank', "noreferrer")
                        }}>
                            screen capture
                        </p>
                    </div>
                </div>
            )
    }, [])


    const coffeeLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://www.buymeacoffee.com/wokad") }}>
                    <img className="donate-img" src="./assets/buymeacoffee.png" />
                    <div className="tooltip-text tooltip-text-100px">donate(寄付)</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://www.buymeacoffee.com/wokad" target="_blank" rel="noopener noreferrer">
                    <img className="donate-img" src="./assets/buymeacoffee.png" />
                    <div className="tooltip-text tooltip-text-100px">
                        donate(寄付)
                    </div>
                </a>
            )
    }, [])


    const titleRow = useMemo(() => {
        if (props.lineNum == 2) {
            return (
                <>
                    <div className="top-title">
                        <span className="title">{props.mainTitle}</span>
                    </div>
                    <div className="top-title">
                        <span className="top-title-version">{props.subTitle}</span>
                        <span className="belongings">
                            {githubLink}
                            {manualLink}
                            {toolLink}
                            {coffeeLink}
                        </span>
                    </div>
                </>

            )

        } else {
            return (
                <div className="top-title">
                    <span className="title">{props.mainTitle}</span>
                    <span className="top-title-version">{props.subTitle}</span>
                    <span className="belongings">
                        {githubLink}
                        {manualLink}
                        {toolLink}
                        {coffeeLink}
                    </span>
                    <span className="belongings">
                    </span>
                </div>
            )
        }
    }, [props.subTitle, props.mainTitle, props.lineNum])

    return titleRow
};
