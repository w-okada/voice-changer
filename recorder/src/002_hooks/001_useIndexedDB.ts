import localForage from "localforage";

export type IndexedDBState = {
    dummy: string
}
export type IndexedDBStateAndMethod = IndexedDBState & {
    setItem: (key: string, value: unknown) => Promise<void>,
    getItem: (key: string) => Promise<unknown>
}

export const useIndexedDB = (): IndexedDBStateAndMethod => {
    localForage.config({
        driver: localForage.INDEXEDDB,
        name: 'app',
        version: 1.0,
        storeName: 'appStorage',
        description: 'appStorage'

    })

    const setItem = async (key: string, value: unknown) => {
        await localForage.setItem(key, value)
    }
    const getItem = async (key: string) => {
        return await localForage.getItem(key)
    }

    return {
        dummy: "",
        setItem,
        getItem
    }
}