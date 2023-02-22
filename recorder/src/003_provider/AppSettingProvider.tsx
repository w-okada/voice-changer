import React, { useContext } from "react";
import { ReactNode } from "react";
import { ApplicationSettingManagerStateAndMethod, useApplicationSettingManager } from "../002_hooks/000_useApplicationSettingManager";
import { IndexedDBStateAndMethod, useIndexedDB } from "../002_hooks/001_useIndexedDB";
import { DeviceManagerStateAndMethod, useDeviceManager } from "../002_hooks/002_useDeviceManager";

import { AppStateStorageStateAndMethod, useAppStateStorage } from "../002_hooks/004_useAppStateStorage";

type Props = {
    children: ReactNode;
};

type AppSettingValue = {
    applicationSetting: ApplicationSettingManagerStateAndMethod
    indexedDBState: IndexedDBStateAndMethod;
    deviceManagerState: DeviceManagerStateAndMethod;
    appStateStorageState: AppStateStorageStateAndMethod
};

const AppSettingContext = React.createContext<AppSettingValue | null>(null);
export const useAppSetting = (): AppSettingValue => {
    const state = useContext(AppSettingContext);
    if (!state) {
        throw new Error("useAppSetting must be used within AppSettingProvider");
    }
    return state;
};

export const AppSettingProvider = ({ children }: Props) => {
    const applicationSetting = useApplicationSettingManager();
    const indexedDBState = useIndexedDB();
    const deviceManagerState = useDeviceManager();
    const appStateStorageState = useAppStateStorage({ applicationSetting: applicationSetting.applicationSetting, indexedDBState })

    const providerValue = {
        applicationSetting,
        indexedDBState,
        deviceManagerState,
        appStateStorageState,
    };
    return <AppSettingContext.Provider value={providerValue}>{children}</AppSettingContext.Provider>;
};
