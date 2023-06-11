import React, { useMemo } from "react";
import { isDesktopApp } from "./const";
import { useAppRoot } from "./001_provider/001_AppRootProvider";
import { useMessageBuilder } from "./hooks/useMessageBuilder";


export type TitleProps = {
    lineNum: number
    mainTitle: string
    subTitle: string
}

export const Title = (props: TitleProps) => {
    const messageBuilderState = useMessageBuilder()
    useMemo(() => {
        messageBuilderState.setMessage(__filename, "github", { "ja": "github", "en": "github" })
        messageBuilderState.setMessage(__filename, "manual", { "ja": "マニュアル", "en": "manual" })
        messageBuilderState.setMessage(__filename, "screenCapture", { "ja": "録画ツール", "en": "Record Screen" })
        messageBuilderState.setMessage(__filename, "support", { "ja": "支援", "en": "Donation" })
    }, [])

    const githubLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://github.com/w-okada/voice-changer") }}>
                    <img src="./assets/icons/github.svg" />
                    <div className="tooltip-text">{messageBuilderState.getMessage(__filename, "github")}</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://github.com/w-okada/voice-changer" target="_blank" rel="noopener noreferrer">
                    <img src="./assets/icons/github.svg" />
                    <div className="tooltip-text">{messageBuilderState.getMessage(__filename, "github")}</div>
                </a>
            )
    }, [])


    const manualLink = useMemo(() => {
        return isDesktopApp() ?
            (
                // @ts-ignore
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md") }}>
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "manual")}</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md" target="_blank" rel="noopener noreferrer">
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "manual")}</div>
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
                            {messageBuilderState.getMessage(__filename, "screenCapture")}
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
                            {messageBuilderState.getMessage(__filename, "screenCapture")}
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
                    <div className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "support")}</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://www.buymeacoffee.com/wokad" target="_blank" rel="noopener noreferrer">
                    <img className="donate-img" src="./assets/buymeacoffee.png" />
                    <div className="tooltip-text tooltip-text-100px">
                        {messageBuilderState.getMessage(__filename, "support")}
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
