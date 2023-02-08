export const fetchTextResource = async (url: string): Promise<string> => {
    const res = await fetch(url, {
        method: "GET"
    });
    const text = res.text()
    return text;
}
