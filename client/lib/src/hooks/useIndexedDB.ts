import localForage from "localforage";
import { useMemo } from "react";
import { ClientType, INDEXEDDB_DB_APP_NAME, INDEXEDDB_DB_NAME } from "../const";

export type UseIndexedDBProps = {
    clientType: ClientType | null
}
export type IndexedDBState = {
    dummy: string
}
export type IndexedDBStateAndMethod = IndexedDBState & {
    setItem: (key: string, value: unknown) => Promise<void>,
    getItem: (key: string) => Promise<unknown>
    removeItem: (key: string) => Promise<void>
    // clearAll: () => Promise<void>
    removeDB: () => Promise<void>
}

export const useIndexedDB = (props: UseIndexedDBProps): IndexedDBStateAndMethod => {
    const clientType = props.clientType || "default"
    localForage.config({
        driver: localForage.INDEXEDDB,
        name: INDEXEDDB_DB_APP_NAME,
        version: 1.0,
        storeName: `${INDEXEDDB_DB_NAME}`,
        description: 'appStorage'

    })

    const setItem = useMemo(() => {
        return async (key: string, value: unknown) => {
            const clientKey = `${clientType}_${key}`
            await localForage.setItem(clientKey, value)
        }
    }, [props.clientType])

    const getItem = useMemo(() => {
        return async (key: string) => {
            const clientKey = `${clientType}_${key}`
            return await localForage.getItem(clientKey)
        }
    }, [props.clientType])

    const removeItem = useMemo(() => {
        return async (key: string) => {
            const clientKey = `${clientType}_${key}`
            console.log("remove key:", clientKey)
            return await localForage.removeItem(clientKey)
        }
    }, [props.clientType])

    const removeDB = useMemo(() => {
        return async () => {
            const keys = await localForage.keys()
            for (const key of keys) {
                console.log("remove key:", key)
                await localForage.removeItem(key)
            }
        }
    }, [props.clientType])


    return {
        dummy: "",
        setItem,
        getItem,
        removeItem,
        removeDB
    }
}