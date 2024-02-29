import React from "react";
import { useMemo } from "react";
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { useAppState } from "../../003_provider/AppStateProvider";

export const CorpusTextArea = () => {
    const { applicationSetting } = useAppSetting()
    const { corpusDataState } = useAppState();
    const { text } = useMemo(() => {
        const corpus = corpusDataState.corpusTextData[applicationSetting.applicationSetting.current_text];
        const text = corpus?.text[applicationSetting.applicationSetting.current_text_index] || "";
        return { text };
    }, [corpusDataState.corpusTextData, applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index]);

    return (
        <div className="card">
            <div className="title">#{applicationSetting.applicationSetting.current_text_index + 1}</div>

            <div className="text">
                <div className="tag">Text</div>
                <div>{text}</div></div>

        </div >
    );
};
