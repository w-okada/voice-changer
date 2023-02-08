import { ApplicationSetting } from "../001_clients_and_managers/000_ApplicationSettingLoader"
import { generateWavNameForLocalStorage } from "../const"
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
        const { micString } = generateWavNameForLocalStorage(prefix, index)
        props.indexedDBState.setItem(micString, userData)
    }

    const loadUserData = async (_title: string, prefix: string, index: number): Promise<FixedUserData | null> => {
        const { micString } = generateWavNameForLocalStorage(prefix, index)
        const obj = await props.indexedDBState.getItem(micString) as FixedUserData
        return obj
    }

    return {
        saveUserData,
        loadUserData,
    }
}
