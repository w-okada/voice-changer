import React, { useMemo } from "react";
import { useGuiState } from "../001_GuiStateProvider";

export type MoreActionAreaProps = {};

export const MoreActionArea = (_props: MoreActionAreaProps) => {
    const { stateControls } = useGuiState();

    const serverIORecorderRow = useMemo(() => {
        const onOpenMergeLabClicked = () => {
            stateControls.showMergeLabCheckbox.updateState(true);
        };
        const onOpenAdvancedSettingClicked = () => {
            stateControls.showAdvancedSettingCheckbox.updateState(true);
        };
        const onOpenGetServerInformationClicked = () => {
            stateControls.showGetServerInformationCheckbox.updateState(true);
        };
        const onOpenGetClientInformationClicked = () => {
            stateControls.showGetClientInformationCheckbox.updateState(true);
        };
        return (
            <>
                <div className="config-sub-area-control left-padding-1">
                    <div className="config-sub-area-control-title">more...</div>
                    <div className="config-sub-area-control-field config-sub-area-control-field-long">
                        <div className="config-sub-area-buttons">
                            <div onClick={onOpenMergeLabClicked} className="config-sub-area-button">
                                Merge Lab
                            </div>
                            <div onClick={onOpenAdvancedSettingClicked} className="config-sub-area-button">
                                Advanced Setting
                            </div>
                            <div onClick={onOpenGetServerInformationClicked} className="config-sub-area-button">
                                Server Info
                            </div>
                            <div onClick={onOpenGetClientInformationClicked} className="config-sub-area-button">
                                Client Info
                            </div>
                        </div>
                    </div>
                </div>
            </>
        );
    }, [stateControls]);

    return <div className="config-sub-area">{serverIORecorderRow}</div>;
};
