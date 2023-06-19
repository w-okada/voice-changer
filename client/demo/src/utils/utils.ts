export const trimfileName = (name: string, length: number) => {
    const trimmedName = name.replace(/^.*[\\\/]/, '')
    if (trimmedName.length > length) {
        return trimmedName.substring(0, length) + "..."
    } else {
        return trimmedName
    }
}

export const checkExtention = (filename: string, acceptExtentions: string[]) => {
    const ext = filename.split('.').pop();
    if (!ext) {
        return false
    }
    return acceptExtentions.includes(ext)
}