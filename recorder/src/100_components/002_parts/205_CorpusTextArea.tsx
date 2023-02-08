import React from "react";
import { useMemo } from "react";
import { useAppSetting } from "../../003_provider/AppSettingProvider";
import { useAppState } from "../../003_provider/AppStateProvider";

export const CorpusTextArea = () => {
    const { applicationSetting } = useAppSetting()
    const { corpusDataState } = useAppState();
    const { text, text_hira } = useMemo(() => {
        const corpus = corpusDataState.corpusTextData[applicationSetting.applicationSetting.current_text];
        const text = corpus?.text[applicationSetting.applicationSetting.current_text_index] || "";
        const text_hira = corpus?.text_hira[applicationSetting.applicationSetting.current_text_index] || "";
        return { text, text_hira };
    }, [corpusDataState.corpusTextData, applicationSetting.applicationSetting.current_text, applicationSetting.applicationSetting.current_text_index]);

    return (
        <div className="card">
            <div className="title">{applicationSetting.applicationSetting.current_text_index + 1}番目</div>

            <div className="text">
                <div className="tag">テキスト</div>
                <div>{text}</div></div>
            <div className="text">
                <div className="tag">読み</div>
                <div>{text_hira}</div></div>
        </div >
    );
};
