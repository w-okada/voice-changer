import React, { Suspense, useMemo } from "react";
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { useAppState } from "../../003_provider/AppStateProvider";

export const CorpusSelector = () => {
    const { applicationSetting } = useAppSetting()
    const { corpusDataState, audioControllerState } = useAppState();

    const audioActive = useMemo(() => {
        return audioControllerState.audioControllerState === "play" || audioControllerState.audioControllerState === "record";
    }, [audioControllerState.audioControllerState]);
    const unsavedRecord = useMemo(() => {
        return audioControllerState.unsavedRecord;
    }, [audioControllerState.unsavedRecord]);


    const options = useMemo(() => {
        const options = Object.keys(corpusDataState.corpusTextData).map((title) => {
            return (
                <option key={title} value={title}>
                    {title}
                </option>
            );
        });
        if (!applicationSetting.applicationSetting.current_text) {
            options.unshift(<option key={"none"} value={"none"}></option>);
        }
        return options;
    }, [corpusDataState.corpusTextData]);

    const selector = useMemo(() => {
        const disabled = audioActive || unsavedRecord
        return (
            <>
                <div className="label">Corpus:</div>
                <div className="selector">
                    <select
                        disabled={disabled ? true : false}
                        defaultValue={applicationSetting.applicationSetting.current_text || ""}
                        onChange={(e) => {
                            applicationSetting.setCurrentText(e.target.value);
                            applicationSetting.setCurrentTextIndex(0);
                        }}
                        className="select"
                    >
                        {options}
                    </select>
                </div>
            </>
        );
    }, [applicationSetting.applicationSetting.current_text, options, audioActive, unsavedRecord]);

    const Wrapper = () => {
        if (Object.keys(corpusDataState.corpusTextData).length === 0) {
            throw new Promise((resolve) => setTimeout(resolve, 1000 * 2));
        }
        return selector;
    };
    return (
        <Suspense fallback={<>loading...</>}>
            <Wrapper></Wrapper>
        </Suspense>
    );
};
