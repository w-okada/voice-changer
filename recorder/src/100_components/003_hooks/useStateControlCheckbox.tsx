import React, { useMemo, useRef } from "react";
import { useEffect } from "react";

export type StateControlCheckbox = {
    trigger: JSX.Element;
    updateState: (newVal: boolean) => void;
    className: string;
};

export const useStateControlCheckbox = (className: string, changeCallback?: (newVal: boolean) => void): StateControlCheckbox => {
    const currentValForTriggerCallbackRef = useRef<boolean>(false);
    // (4) トリガチェックボックス
    const callback = useMemo(() => {
        console.log("generate callback function", className);
        return (newVal: boolean) => {
            if (!changeCallback) {
                return;
            }
            //  値が同じときはスルー (== 初期値(undefined)か、値が違ったのみ発火)
            if (currentValForTriggerCallbackRef.current === newVal) {
                return;
            }
            // 初期値(undefined)か、値が違ったのみ発火
            currentValForTriggerCallbackRef.current = newVal;
            changeCallback(currentValForTriggerCallbackRef.current);
        };
    }, []);
    const trigger = useMemo(() => {
        if (changeCallback) {
            return (
                <input
                    type="checkbox"
                    className={`${className} state-control-checkbox rotate-button`}
                    id={`${className}`}
                    onChange={(e) => {
                        callback(e.target.checked);
                    }}
                />
            );
        } else {
            return <input type="checkbox" className={`${className} state-control-checkbox rotate-button`} id={`${className}`} />;
        }
    }, []);

    useEffect(() => {
        const checkboxes = document.querySelectorAll(`.${className}`);
        // (1) On/Off同期
        checkboxes.forEach((x) => {
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore
            x.onchange = (ev) => {
                updateState(ev.target.checked);
            };
        });
        // (2) 全エレメントoff
        const removers = document.querySelectorAll(`.${className}-remover`);
        removers.forEach((x) => {
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-ignore
            x.onclick = (ev) => {
                if (ev.target.className.indexOf(`${className}-remover`) > 0) {
                    updateState(false);
                }
            };
        });
    }, []);

    // (3) ステート変更
    const updateState = useMemo(() => {
        return (newVal: boolean) => {
            const currentCheckboxes = document.querySelectorAll(`.${className}`);
            currentCheckboxes.forEach((y) => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-ignore
                y.checked = newVal;
            });
            if (changeCallback) {
                callback(newVal);
            }
        };
    }, []);

    return {
        trigger,
        updateState,
        className,
    };
};
