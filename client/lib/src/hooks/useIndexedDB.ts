import localForage from "localforage";
import { useMemo } from "react";
import { ClientType, INDEXEDDB_DB_APP_NAME, INDEXEDDB_DB_NAME } from "../const";

export type UseIndexedDBProps = {
    clientType: ClientType
}
export type IndexedDBState = {
    dummy: string
}
export type IndexedDBStateAndMethod = IndexedDBState & {
    setItem: (key: string, value: unknown) => Promise<void>,
    getItem: (key: string) => Promise<unknown>
    removeItem: (key: string) => Promise<void>
    // clearAll: () => Promise<void>
}

export const useIndexedDB = (props: UseIndexedDBProps): IndexedDBStateAndMethod => {
    localForage.config({
        driver: localForage.INDEXEDDB,
        name: INDEXEDDB_DB_APP_NAME,
        version: 1.0,
        storeName: `${INDEXEDDB_DB_NAME}_${props.clientType}`,
        description: 'appStorage'

    })

    const setItem = useMemo(() => {
        return async (key: string, value: unknown) => {
            await localForage.setItem(key, value)
        }
    }, [])
    const getItem = useMemo(() => {
        return async (key: string) => {
            return await localForage.getItem(key)
        }
    }, [])

    const removeItem = useMemo(() => {
        return async (key: string) => {
            return await localForage.removeItem(key)
        }
    }, [])


    return {
        dummy: "",
        setItem,
        getItem,
        removeItem,
    }
}