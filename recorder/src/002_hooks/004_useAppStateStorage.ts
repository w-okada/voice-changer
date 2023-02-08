import { ApplicationSetting } from "../001_clients_and_managers/000_ApplicationSettingLoader"
import { generateDataNameForLocalStorage } from "../const"
import { IndexedDBStateAndMethod } from "./001_useIndexedDB"
import { FixedUserData } from "./013_useAudioControllerState"
export type UseAppStateStorageProps = {
    applicationSetting: ApplicationSetting | null
    indexedDBState: IndexedDBStateAndMethod
}

export type AppStateStorageState = {

}
export type AppStateStorageStateAndMethod = AppStateStorageState & {
    saveUserData: (title: string, prefix: string, index: number, userData: FixedUserData) => void
    loadUserData: (title: string, prefix: string, index: number) => Promise<FixedUserData | null>
}

export const useAppStateStorage = (props: UseAppStateStorageProps): AppStateStorageStateAndMethod => {
    const saveUserData = async (_title: string, prefix: string, index: number, userData: FixedUserData) => {
        const { dataName } = generateDataNameForLocalStorage(prefix, index)
        props.indexedDBState.setItem(dataName, userData)
    }

    const loadUserData = async (_title: string, prefix: string, index: number): Promise<FixedUserData | null> => {
        const { dataName } = generateDataNameForLocalStorage(prefix, index)
        const obj = await props.indexedDBState.getItem(dataName) as FixedUserData
        return obj
    }

    return {
        saveUserData,
        loadUserData,
    }
}
