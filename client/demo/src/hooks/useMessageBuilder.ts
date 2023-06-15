import { useRef } from "react"

export type Message = {
    file: string,
    id: string,
    message: { [lang: string]: string }
}

export type MessageBuilderStateAndMethod = {
    setMessage: (file: string, id: string, message: { [lang: string]: string }) => void
    getMessage: (file: string, id: string) => string
}

export const useMessageBuilder = (): MessageBuilderStateAndMethod => {
    const messagesRef = useRef<Message[]>([])

    const setMessage = (file: string, id: string, message: { [lang: string]: string }) => {
        if (messagesRef.current.find(x => { return x.file == file && x.id == id })) {
            console.warn("duplicate message is registerd", file, id, message)
        } else {
            messagesRef.current.push({ file, id, message })
        }
    }
    const getMessage = (file: string, id: string) => {
        let lang = window.navigator.language
        if (lang != "ja") {
            lang = "en"
        }

        return messagesRef.current.find(x => { return x.file == file && x.id == id })?.message[lang] || "unknwon message"
    }
    return {
        setMessage,
        getMessage
    }
}