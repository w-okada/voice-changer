import React, { useMemo } from "react";
import { useAppState } from "../../003_provider/AppStateProvider";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useAppSetting } from "../../003_provider/AppSettingProvider";

export const TextIndexSelector = () => {
    const { applicationSetting } = useAppSetting()
    const { corpusDataState, audioControllerState } = useAppState();

    const audioActive = useMemo(() => {
        return audioControllerState.audioControllerState === "play" || audioControllerState.audioControllerState === "record";
    }, [audioControllerState.audioControllerState]);
    const unsavedRecord = useMemo(() => {
        return audioControllerState.unsavedRecord;
    }, [audioControllerState.unsavedRecord]);

    const prevButton = useMemo(() => {
        let className = "";
        let prevIndex = () => { };
        if (applicationSetting.applicationSetting.current_text_index === 0 || audioActive || unsavedRecord) {
            className = "index-selector-button disable";
        } else {
            className = "index-selector-button";
            prevIndex = () => {
                applicationSetting.setCurrentTextIndex(applicationSetting.applicationSetting.current_text_index - 1);
            };
        }
        return (
            <div className={className} onClick={prevIndex}>
                <FontAwesomeIcon icon={["fas", "angle-left"]} size="1x" />
            </div>
        );
    }, [applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index, audioActive, unsavedRecord]);


    const nextButton = useMemo(() => {
        const corpus = corpusDataState.corpusTextData[applicationSetting.applicationSetting.current_text];
        if (!corpus) {
            return <></>
        }

        let className = "";
        let nextIndex = () => { };
        const length = corpus.text.length;
        if (applicationSetting.applicationSetting.current_text_index === length - 1 || audioActive || unsavedRecord) {
            className = "index-selector-button disable";
        } else {
            className = "index-selector-button";
            nextIndex = () => {
                applicationSetting.setCurrentTextIndex(applicationSetting.applicationSetting.current_text_index + 1);
            };
        }
        return (
            <div className={className} onClick={nextIndex}>
                <FontAwesomeIcon icon={["fas", "angle-right"]} size="1x" />
            </div>
        );
    }, [corpusDataState.corpusTextData, applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index, audioActive, unsavedRecord]);

    const indexText = useMemo(() => {
        const corpus = corpusDataState.corpusTextData[applicationSetting.applicationSetting.current_text];
        if (!corpus) {
            return <></>
        }

        const length = corpus.text.length;
        const text = `${applicationSetting.applicationSetting.current_text_index + 1}/${length}`;
        return <div className="label">{text}</div>;
    }, [corpusDataState.corpusTextData, applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index]);
    return (
        <>
            {prevButton}
            {indexText}
            {nextButton}
        </>
    );
};
