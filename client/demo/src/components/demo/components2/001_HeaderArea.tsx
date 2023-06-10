import React, { useMemo } from "react";
import { INDEXEDDB_KEY_AUDIO_OUTPUT, INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, isDesktopApp } from "../../../const";
import { useGuiState } from "../001_GuiStateProvider";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useIndexedDB } from "@dannadori/voice-changer-client-js";


export type HeaderAreaProps = {
    mainTitle: string
    subTitle: string
}

export const HeaderArea = (props: HeaderAreaProps) => {
    const { appGuiSettingState, setClientType } = useAppRoot()
    const { clientSetting, clearSetting } = useAppState()
    const { setIsConverting, isConverting } = useGuiState()

    const clientType = appGuiSettingState.appGuiSetting.id
    const { removeItem } = useIndexedDB({ clientType: clientType })
    const { setItem } = useIndexedDB({ clientType: null })


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
                <span className="link tooltip" onClick={() => { window.electronAPI.openBrowser("https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md") }}>
                    <img src="./assets/icons/help-circle.svg" />
                    <div className="tooltip-text">manual</div>
                </span>
            )
            :
            (
                <a className="link tooltip" href="https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md" target="_blank" rel="noopener noreferrer">
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

    const headerArea = useMemo(() => {
        const onClearSettingClicked = async () => {
            await clearSetting()
            await removeItem(INDEXEDDB_KEY_AUDIO_OUTPUT)
            location.reload()
        }
        const onReloadClicked = async () => {
            location.reload()
        }
        const onReselectVCClicked = async () => {
            setIsConverting(false)
            if (isConverting) {
                await clientSetting.stop()
                setIsConverting(false)
            }
            setItem(INDEXEDDB_KEY_DEFAULT_MODEL_TYPE, "null")
            setClientType(null)
            appGuiSettingState.clearAppGuiSetting()

        }

        return (
            <div className="headerArea">
                <div className="title1">
                    <span className="title">{props.mainTitle}</span>
                    <span className="title-version">{props.subTitle}</span>
                    <span className="title-version-number">{appGuiSettingState.version}</span>
                    <span className="title-version-number">{appGuiSettingState.edition}</span>
                </div>
                <div className="icons">
                    <span className="belongings">
                        {githubLink}
                        {manualLink}
                        {toolLink}
                        {coffeeLink}
                        {/* {licenseButton} */}
                    </span>
                    <span className="belongings">
                        <div className="belongings-button" onClick={onClearSettingClicked}>clear setting</div>
                        <div className="belongings-button" onClick={onReloadClicked}>reload</div>
                        <div className="belongings-button" onClick={onReselectVCClicked}>select vc</div>
                    </span>
                </div>

            </div>
        )
    }, [props.subTitle, props.mainTitle, appGuiSettingState.version, appGuiSettingState.edition])

    return headerArea
};
